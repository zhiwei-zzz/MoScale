import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# automatically import fused operators
dropout_add_layer_norm = fused_mlp_func = memory_efficient_attention = flash_attn_func = None
from flash_attn import flash_attn_func                  # q, k, or v: BLHc, ret: BLHc
from flash_attn import flash_attn_varlen_kvpacked_func  # qkv: N3Hc, ret: NHc
try:
    from flash_attn.ops.layer_norm import dropout_add_layer_norm
    from flash_attn.ops.fused_dense import fused_mlp_func
except ImportError: pass
# automatically import faster attention implementations
try: from xformers.ops import memory_efficient_attention
except ImportError: pass
try: from flash_attn import flash_attn_func              # qkv: BLHc, ret: BLHcq
except ImportError: pass
try: from torch.nn.functional import scaled_dot_product_attention as slow_attn    # q, k, v: BHLc
except ImportError:
    def slow_attn(query, key, value, scale: float, attn_mask=None, dropout_p=0.0):
        attn = query.mul(scale) @ key.transpose(-2, -1) # BHLc @ BHcL => BHLL
        if attn_mask is not None: attn.add_(attn_mask)
        return (F.dropout(attn.softmax(dim=-1), p=dropout_p, inplace=True) if dropout_p > 0 else attn.softmax(dim=-1)) @ value

try:
    from flash_attn.ops.rms_norm import rms_norm as rms_norm_impl
except ImportError:
    def rms_norm_impl(x, weight, epsilon):
        return (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True).add_(epsilon))) * weight


def get_dropout_layer(p):
    return nn.Dropout(p, inplace=True) if p > 0 else nn.Identity()


class FastRMSNorm(nn.Module):
    def __init__(self, C, eps=1e-6, elementwise_affine=True):
        super().__init__()
        self.C = C
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.ones(C))
        else:
            self.register_buffer('weight', torch.ones(C))

    def forward(self, x):
        src_type = x.dtype
        return rms_norm_impl(x.float(), self.weight, epsilon=self.eps).to(src_type)

    def extra_repr(self) -> str:
        return f'C={self.C}, eps={self.eps:g}, elementwise_affine={self.elementwise_affine}'


class CrossAttention(nn.Module):
    def __init__(
        self, for_attn_pool=False, embed_dim=768, kv_dim=4096, num_heads=12,
        proj_drop=0., cos_attn=False,
    ):
        super().__init__()
        self.for_attn_pool = for_attn_pool
        self.embed_dim = embed_dim
        self.kv_dim = kv_dim
        assert embed_dim % num_heads == 0
        self.num_heads, self.head_dim = num_heads, embed_dim // num_heads  # =64
        self.scale = 1 / math.sqrt(self.head_dim)

        if for_attn_pool:
            q = torch.empty(1, self.num_heads, self.head_dim)
            nn.init.trunc_normal_(q, mean=0, std=math.sqrt(1 / embed_dim / 3))
            self.mat_q = nn.Parameter(q)
        else:
            self.mat_q = nn.Linear(embed_dim, embed_dim, bias=True)
        self.mat_kv = nn.Linear(kv_dim, embed_dim*2, bias=False)
        self.v_bias = nn.Parameter(torch.zeros(embed_dim))
        self.register_buffer('zero_k_bias', torch.zeros(embed_dim))

        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = get_dropout_layer(proj_drop)

    def forward(self, q, ca_kv):
        """
        :param q: shaped as (batch, seq_len, Q_dim)
        :param ca_kv: contains several vectors, each of which is shaped as (len_i, KV_dim). We have [len_1xKV_dim, len_2xKV_dim, len_3xKV_dim, ...] and lens == [len_1, len_2, len_3, ...]
            - kv_compact: shaped as (sum(lens), KV_dim)
            - cu_seqlens_k: cumulated sum of lens
            - max_seqlen_k: int, max(lens)
        NOTE: seq_len (num of Qs) can reach 10k;  but len_i (num of KVs) must <= 256

        :return: shaped as (batch, seq_len, Q_dim)
        """
        kv_compact, cu_seqlens_k, max_seqlen_k = ca_kv
        N = kv_compact.shape[0]

        kv_compact = F.linear(kv_compact, weight=self.mat_kv.weight, bias=torch.cat((self.zero_k_bias, self.v_bias))).view(N, 2, self.num_heads, self.head_dim) # NC => N2Hc

        if not self.for_attn_pool:
            B, Lq = q.shape[:2]
            q_compact = self.mat_q(q).view(-1, self.num_heads, self.head_dim)
        else:
            B = cu_seqlens_k.shape[0] - 1
            Lq = 1
            q_compact = self.mat_q.repeat(B, 1, 1).to(dtype=kv_compact.dtype)

        q_compact = q_compact.contiguous()
        kv_compact = kv_compact.contiguous()

        cu_seqlens_q = torch.arange(0, Lq * (B+1), Lq, dtype=torch.int32, device=q_compact.device)
        if q_compact.dtype == torch.float32:    # todo: fp16 or bf16?
            oup = flash_attn_varlen_kvpacked_func(q=q_compact.to(dtype=torch.bfloat16), kv=kv_compact.to(dtype=torch.bfloat16), cu_seqlens_q=cu_seqlens_q, cu_seqlens_k=cu_seqlens_k, max_seqlen_q=Lq, max_seqlen_k=max_seqlen_k, dropout_p=0, softmax_scale=self.scale).reshape(B, Lq, -1)
            oup = oup.float()
        else:
            oup = flash_attn_varlen_kvpacked_func(q=q_compact, kv=kv_compact, cu_seqlens_q=cu_seqlens_q, cu_seqlens_k=cu_seqlens_k, max_seqlen_q=Lq, max_seqlen_k=max_seqlen_k, dropout_p=0, softmax_scale=self.scale).reshape(B, Lq, -1)

        return self.proj_drop(self.proj(oup))

    def extra_repr(self) -> str:
        return f'Cq={self.embed_dim}, Ckv={self.kv_dim}'


class MultiQueryTextPool(nn.Module):
    """
    Produce M (=first_l) distinct prefix tokens by attending to the T5 token sequence.
    Uses your existing CrossAttention in 'kv-packed varlen' mode.
    """
    def __init__(self, Ct5: int, D: int, M: int):
        super().__init__()
        self.Ct5, self.D, self.M = Ct5, D, M
        # pick head_dim like your TextAttentivePool
        head_dim = 64 if D > 4096 else 128
        self.num_heads = D // head_dim
        assert self.num_heads * head_dim == D, "D must be divisible by head_dim"

        # Learnable query tokens (M queries)
        q = torch.empty(M, D)
        nn.init.trunc_normal_(q, mean=0.0, std=math.sqrt(1.0 / (D * 3)))
        self.q_tokens = nn.Parameter(q)  # [M, D]

        # Reuse your CrossAttention (not for_attn_pool, since we provide q explicitly)
        self.ca = CrossAttention(
            for_attn_pool=False, embed_dim=D, kv_dim=Ct5, num_heads=self.num_heads, proj_drop=0., cos_attn=False
        )

    def forward(self, ca_kv):
        """
        ca_kv: (kv_compact, cu_seqlens_k, max_seqlen_k) as in your code
        returns: [B, M, D]
        """
        kv_compact, cu_seqlens_k, _ = ca_kv
        B = cu_seqlens_k.shape[0] - 1
        q = self.q_tokens.unsqueeze(0).expand(B, self.M, self.D).contiguous()  # [B, M, D]
        return self.ca(q, ca_kv)  # [B, M, D]


def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):    # taken from timm
    if drop_prob == 0. or not training: return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):  # taken from timm
    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self):
        return f'(drop_prob=...)'


class FFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0., fused_if_available=True):
        super().__init__()
        self.fused_mlp_func = fused_mlp_func if fused_if_available else None
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU(approximate='tanh')
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop, inplace=True) if drop > 0 else nn.Identity()

    def forward(self, x):
        if self.fused_mlp_func is not None:
            return self.drop(self.fused_mlp_func(
                x=x, weight1=self.fc1.weight, weight2=self.fc2.weight, bias1=self.fc1.bias, bias2=self.fc2.bias,
                activation='gelu_approx', save_pre_act=self.training, return_residual=False, checkpoint_lvl=0,
                heuristic=0, process_group=None,
            ))
        else:
            return self.drop(self.fc2( self.act(self.fc1(x)) ))

    def extra_repr(self) -> str:
        return f'fused_mlp_func={self.fused_mlp_func is not None}'


class SelfAttention(nn.Module):
    def __init__(self, block_idx, embed_dim=768, num_heads=12,
                 attn_drop=0., proj_drop=0., attn_l2_norm=False,
                 flash_if_available=True, use_rope=False):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.block_idx, self.num_heads, self.head_dim = block_idx, num_heads, embed_dim // num_heads
        self.attn_l2_norm = attn_l2_norm
        if self.attn_l2_norm:
            self.scale = 1
            self.scale_mul_1H11 = nn.Parameter(torch.full((1, self.num_heads, 1, 1), 4.0).log(), requires_grad=True)
            self.max_scale_mul = torch.log(torch.tensor(100.)).item()
        else:
            self.scale = 0.25 / math.sqrt(self.head_dim)

        self.mat_qkv = nn.Linear(embed_dim, embed_dim * 3, bias=False)
        self.q_bias, self.v_bias = nn.Parameter(torch.zeros(embed_dim)), nn.Parameter(torch.zeros(embed_dim))
        self.register_buffer('zero_k_bias', torch.zeros(embed_dim))

        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(proj_drop, inplace=True) if proj_drop > 0 else nn.Identity()
        self.attn_drop: float = attn_drop
        self.using_flash = flash_if_available and (flash_attn_func is not None)
        self.using_xform  = flash_if_available and (memory_efficient_attention is not None)

        # ---- cache state ----
        self.caching, self.cached_k, self.cached_v = False, None, None   # append mode
        self.cache_mode = None        # None | "append" | "scatter"
        self.k_cache_full = None      # [B, Ltot, H, d]  (BLHc)  <-- changed layout
        self.v_cache_full = None      # [B, Ltot, H, d]
        self.cache_len_full = None
        self.use_rope = use_rope

    # --------- helpers ----------
    @torch.no_grad()
    def _apply_rope_per_pos_inplace(self, t, rope_batch, positions, layout_is_blhc: bool):
        """
        t: BLHc if layout_is_blhc else BHLc. RoPE applied in-place on even/odd dims.
        rope_batch: [B,2,Lpad,half] or [B,2,Lpad,half] wrapped in [...,0,0,0]
        positions:  [B, Ls] absolute positions
        """
        if rope_batch.dim() == 7:
            rope = rope_batch[:, :, 0, 0, 0]  # [B,2,Lpad,half]
        else:
            rope = rope_batch

        if layout_is_blhc:
            T = t.permute(0, 2, 1, 3)  # -> BHLc
        else:
            T = t                     # BHLc

        B, H, Ls, D = T.shape
        half = D // 2
        idx = positions[..., None].expand(B, Ls, half)  # [B,Ls,half]
        cos = rope[:, 0].gather(dim=1, index=idx).unsqueeze(1).to(T.dtype)  # [B,1,Ls,half]
        sin = rope[:, 1].gather(dim=1, index=idx).unsqueeze(1).to(T.dtype)

        device_type = T.device.type if T.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            t0, t1 = T[..., ::2], T[..., 1::2]  # views
            rt0 = t0 * cos - t1 * sin
            rt1 = t0 * sin + t1 * cos
            t0.copy_(rt0); t1.copy_(rt1)

        if layout_is_blhc:
            return T.permute(0, 2, 1, 3)  # back to BLHc
        return T

    def _apply_rope_from_batch(self, q, k, rope_batch, using_flash, start, L):
        """
        q,k: either BLHc (flash/xformers) or BHLc (non-flash). Returns same layout.
        rope_batch: [B, 2, 1, 1, 1, L_pad, half]  or  [B, 2, L_pad, half]
        start: int offset into rope cache (e.g., cached length)
        L:     current chunk length to rotate
        """
        # normalize rope cache to [B, 2, L_pad, half]
        if rope_batch.dim() == 7:
            rope = rope_batch[:, :, 0, 0, 0]           # -> [B, 2, L_pad, half]
        elif rope_batch.dim() == 4:
            rope = rope_batch
        else:
            raise ValueError(f"Unexpected rope_batch shape: {tuple(rope_batch.shape)}")

        # Convert q,k to BHLc for rotation
        if using_flash or self.using_xform:
            qh, kh = q.permute(0, 2, 1, 3), k.permute(0, 2, 1, 3)  # BHLc
        else:
            qh, kh = q, k

        B, H, L_now, D = qh.shape
        half = D // 2
        # Slice RoPE window [start : start+L]
        if start + L_now > rope.shape[2]:
            raise ValueError(f"RoPE slice OOB: need {start+L_now}, have {rope.shape[2]}")
        # cos/sin -> [B,1,L,half] for broadcast across heads
        cos = rope[:, 0, start:start+L_now, :].unsqueeze(1).to(qh.dtype)   # B 1 L half
        sin = rope[:, 1, start:start+L_now, :].unsqueeze(1).to(qh.dtype)

        # Pairwise rotate (even, odd)
        device_type = qh.device.type if qh.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            q0, q1 = qh[..., ::2], qh[..., 1::2]
            k0, k1 = kh[..., ::2], kh[..., 1::2]
            rq0 = q0 * cos - q1 * sin; rq1 = q0 * sin + q1 * cos
            rk0 = k0 * cos - k1 * sin; rk1 = k0 * sin + k1 * cos
            qh = torch.empty_like(qh)
            kh = torch.empty_like(kh)
            qh[..., ::2], qh[..., 1::2] = rq0, rq1
            kh[..., ::2], kh[..., 1::2] = rk0, rk1

        # Return to original layout
        if using_flash or self.using_xform:
            q, k = qh.permute(0, 2, 1, 3), kh.permute(0, 2, 1, 3)  # back to BLHc
        else:
            q, k = qh, kh
        return q, k

    def set_cache(self, mode, total_len=None, batch_size=None, device=None, dtype=None):
        """
        Scatter mode now uses BLHc layout for cache: [B, Ltot, H, d]
        """
        assert mode in (None, "append", "scatter")
        self.cache_mode = mode
        if mode != "append":
            self.caching, self.cached_k, self.cached_v = False, None, None
        if mode == "scatter":
            assert total_len is not None, "set_cache('scatter', total_len=...) required"
            self.cache_len_full = total_len
            if (self.k_cache_full is None) or (self.k_cache_full.shape[1] != total_len):
                assert batch_size is not None, "Provide batch_size on first scatter init"
                H, d = self.num_heads, self.head_dim
                dev = device if device is not None else (self.k_cache_full.device if self.k_cache_full is not None else "cpu")
                dt  = dtype  if dtype  is not None else torch.float16
                self.k_cache_full = torch.zeros(batch_size, total_len, H, d, device=dev, dtype=dt)
                self.v_cache_full = torch.zeros(batch_size, total_len, H, d, device=dev, dtype=dt)

    # ------------- forward -------------
    def forward(self,
                x,
                attn_bias,
                rope_batch,
                *,
                kv_update_indices=None,  # [B, Lupd] absolute write positions
                q_indices=None,          # [B, Lq]   absolute read positions (defaults to kv_update_indices)
                return_kv=False,         # if True, also return (k, v) in BLHc layout
                ):
        B, L, C = x.shape
        qkv = F.linear(x, self.mat_qkv.weight, torch.cat((self.q_bias, self.zero_k_bias, self.v_bias))) \
                .view(B, L, 3, self.num_heads, self.head_dim)
        main_type = qkv.dtype

        using_flash = self.using_flash and (attn_bias is None) and (qkv.dtype != torch.float32)

        if using_flash or self.using_xform:
            q, k, v = qkv.unbind(dim=2)  # BLHc
            layout_is_blhc = True
            dim_cat = 1
        else:
            q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(dim=0)  # BHLc
            layout_is_blhc = False
            dim_cat = 2

        # optional norm
        if self.attn_l2_norm:
            scale_mul = self.scale_mul_1H11.clamp_max(self.max_scale_mul).exp()
            if layout_is_blhc:
                scale_mul = scale_mul.transpose(1, 2)
            q = F.normalize(q, dim=-1).mul(scale_mul)
            k = F.normalize(k, dim=-1)

        # ---- Case A: one-pass no cache ----
        if self.cache_mode is None:
            if self.use_rope:
                assert rope_batch is not None
                q, k = self._apply_rope_from_batch(q, k, rope_batch, layout_is_blhc, start=0, L=L)
            # optionally capture K/V in BLHc layout before attention
            kv_out = None
            if return_kv:
                if layout_is_blhc:
                    kv_out = (k.clone(), v.clone())  # already BLHc
                else:
                    kv_out = (k.permute(0, 2, 1, 3).clone(), v.permute(0, 2, 1, 3).clone())  # BHLc -> BLHc
            dropout_p = self.attn_drop if self.training else 0.0
            if using_flash:
                out = flash_attn_func(q.to(main_type), k.to(main_type), v.to(main_type),
                                      dropout_p=dropout_p, softmax_scale=self.scale).view(B, L, C)
            elif self.using_xform:
                out = memory_efficient_attention(q.to(main_type), k.to(main_type), v.to(main_type),
                    attn_bias=None if attn_bias is None else attn_bias.to(main_type).expand(B, self.num_heads, -1, -1),
                    p=dropout_p, scale=self.scale).view(B, L, C)
            else:
                out = slow_attn(query=q, key=k, value=v, scale=self.scale,
                                attn_mask=attn_bias, dropout_p=dropout_p).transpose(1, 2).reshape(B, L, C)
            result = self.proj_drop(self.proj(out))
            if return_kv:
                return result, kv_out
            return result

        # ---- Case B: append-mode cache ----
        if self.cache_mode == "append":
            if self.use_rope:
                start = 0
                if self.caching and (self.cached_k is not None):
                    start = self.cached_k.shape[dim_cat]
                q, k = self._apply_rope_from_batch(q, k, rope_batch, layout_is_blhc, start=start, L=L)
            if self.caching:
                if self.cached_k is None:
                    self.cached_k, self.cached_v = k, v
                else:
                    k = self.cached_k = torch.cat((self.cached_k, k), dim=dim_cat)
                    v = self.cached_v = torch.cat((self.cached_v, v), dim=dim_cat)

            dropout_p = self.attn_drop if self.training else 0.0
            if using_flash:
                out = flash_attn_func(q.to(main_type), k.to(main_type), v.to(main_type),
                                      dropout_p=dropout_p, softmax_scale=self.scale).view(B, L, C)
            elif self.using_xform:
                out = memory_efficient_attention(q.to(main_type), k.to(main_type), v.to(main_type),
                                                 attn_bias=None, p=dropout_p, scale=self.scale).view(B, L, C)
            else:
                out = slow_attn(query=q, key=k, value=v, scale=self.scale,
                                attn_mask=None, dropout_p=dropout_p).transpose(1, 2).reshape(B, L, C)
            return self.proj_drop(self.proj(out))

        # ---- Case C: scatter-mode cache (fast path) ----
        assert self.cache_mode == "scatter"
        assert kv_update_indices is not None
        if q_indices is None:
            q_indices = kv_update_indices

        # 1) Gather Q to q_indices first (so RoPE lengths match)
        Lq = q_indices.shape[1]
        if layout_is_blhc:
            if q.shape[1] != Lq:
                q = q.gather(dim=1, index=q_indices[..., None, None].expand(B, Lq, q.shape[2], q.shape[3]))
        else:
            if q.shape[2] != Lq:
                q = q.gather(dim=2, index=q_indices[:, None, :, None].expand(B, q.shape[1], Lq, q.shape[3]))

        # 2) RoPE in-place on gathered Q and on K for kv_update_indices
        if self.use_rope:
            assert rope_batch is not None, "RoPE requested but rope_batch is None"
            k = self._apply_rope_per_pos_inplace(k, rope_batch, kv_update_indices, layout_is_blhc)
            q = self._apply_rope_per_pos_inplace(q, rope_batch, q_indices,      layout_is_blhc)

        # 3) Ensure K/V updates are BLHc once (to avoid permute churn)
        if layout_is_blhc:
            k_sc_blhc, v_sc_blhc = k, v                         # BLHc
        else:
            k_sc_blhc, v_sc_blhc = k.permute(0, 2, 1, 3), v.permute(0, 2, 1, 3)  # BHLc->BLHc

        # 4) Scatter updates with batched scatter_ along length dim (no advanced indexing)
        Lupd = kv_update_indices.shape[1]
        idx_len = kv_update_indices[..., None, None].expand(B, Lupd, self.num_heads, self.head_dim)  # [B,Lupd,H,d]
        self.k_cache_full.scatter_(dim=1, index=idx_len, src=k_sc_blhc)
        self.v_cache_full.scatter_(dim=1, index=idx_len, src=v_sc_blhc)

        # 5) Visible prefix slice (prevents future leakage)
        Lvis = int(kv_update_indices.max().item()) + 1
        k_full = self.k_cache_full[:, :Lvis].to(dtype=main_type).contiguous()  # BLHc
        v_full = self.v_cache_full[:, :Lvis].to(dtype=main_type).contiguous()  # BLHc

        # 6) Attention (Flash on fast path; slow path permutes once)
        dropout_p = self.attn_drop if self.training else 0.0
        C = self.num_heads * self.head_dim
        if using_flash:
            out = flash_attn_func(q.to(main_type), k_full, v_full,
                                  dropout_p=dropout_p, softmax_scale=self.scale).view(B, Lq, C)
        elif self.using_xform:
            out = memory_efficient_attention(q.to(main_type), k_full, v_full,
                                             attn_bias=None, p=dropout_p, scale=self.scale).view(B, Lq, C)
        else:
            # permute once for slow path
            k_bhlc = k_full.permute(0, 2, 1, 3).contiguous()
            v_bhlc = v_full.permute(0, 2, 1, 3).contiguous()
            q_bhlc = q if not layout_is_blhc else q.permute(0, 2, 1, 3).contiguous()
            out = slow_attn(query=q_bhlc, key=k_bhlc, value=v_bhlc, scale=self.scale,
                            attn_mask=None, dropout_p=dropout_p) \
                    .transpose(1, 2).reshape(B, Lq, C)

        return self.proj_drop(self.proj(out))

    def forward_with_prefix_cache(self, x_tail, rope_batch, rope_start, prefix_kv):
        """
        Attend tail tokens to (cached_prefix + tail) K/V, with no attention mask.

        Args:
            x_tail:     [B, L_tail, C]  — hidden states for the current-scale tokens only
            rope_batch: RoPE cache (same format as forward())
            rope_start: int — RoPE position offset for tail tokens (= prefix_len)
            prefix_kv:  (k_pfx, v_pfx) each [B, L_pfx, H, d] in BLHc layout

        Returns:
            (output [B, L_tail, C],  (k_tail, v_tail) each [B, L_tail, H, d] in BLHc)
        """
        B, L_tail, C = x_tail.shape
        qkv = F.linear(x_tail, self.mat_qkv.weight,
                        torch.cat((self.q_bias, self.zero_k_bias, self.v_bias))) \
                .view(B, L_tail, 3, self.num_heads, self.head_dim)
        main_type = qkv.dtype

        # Always use BLHc layout for this path (flash attn preferred)
        q, k, v = qkv.unbind(dim=2)  # each [B, L_tail, H, d]

        if self.attn_l2_norm:
            scale_mul = self.scale_mul_1H11.clamp_max(self.max_scale_mul).exp()
            scale_mul_blhc = scale_mul.transpose(1, 2)  # [1,1,H,1]
            q = F.normalize(q, dim=-1).mul(scale_mul_blhc)
            k = F.normalize(k, dim=-1)

        # Apply RoPE to tail Q/K starting at rope_start
        if self.use_rope:
            assert rope_batch is not None
            q, k = self._apply_rope_from_batch(q, k, rope_batch,
                                               using_flash=True,  # BLHc layout
                                               start=rope_start, L=L_tail)

        # Save tail K/V for potential future use
        k_tail_out = k.clone()
        v_tail_out = v.clone()

        # Concatenate cached prefix K/V with tail K/V
        k_pfx, v_pfx = prefix_kv  # each [B, L_pfx, H, d]
        k_full = torch.cat([k_pfx.to(dtype=main_type), k], dim=1)  # [B, L_pfx + L_tail, H, d]
        v_full = torch.cat([v_pfx.to(dtype=main_type), v], dim=1)

        # Attention: Q(tail) attends to K/V(prefix + tail), no mask needed
        dropout_p = self.attn_drop if self.training else 0.0
        if self.using_flash and (main_type != torch.float32):
            out = flash_attn_func(q.to(main_type), k_full, v_full,
                                  dropout_p=dropout_p, softmax_scale=self.scale) \
                      .view(B, L_tail, C)
        elif self.using_xform:
            out = memory_efficient_attention(q.to(main_type), k_full, v_full,
                                             attn_bias=None, p=dropout_p,
                                             scale=self.scale).view(B, L_tail, C)
        else:
            # slow path: permute to BHLc
            q_bhlc = q.permute(0, 2, 1, 3)
            k_bhlc = k_full.permute(0, 2, 1, 3)
            v_bhlc = v_full.permute(0, 2, 1, 3)
            out = slow_attn(query=q_bhlc, key=k_bhlc, value=v_bhlc,
                            scale=self.scale, attn_mask=None,
                            dropout_p=dropout_p) \
                      .transpose(1, 2).reshape(B, L_tail, C)

        return self.proj_drop(self.proj(out)), (k_tail_out, v_tail_out)


class AdaLNSelfAttn(nn.Module):
    def __init__(
        self, block_idx, last_drop_p, embed_dim, cond_dim, shared_aln: bool, norm_layer,
        num_heads, mlp_ratio=4., drop=0., attn_drop=0., drop_path=0., attn_l2_norm=False,
        flash_if_available=False, fused_if_available=True, use_rope=False,
        use_crossattn=False, cos_attn=False, kv_dim=0
    ):
        super(AdaLNSelfAttn, self).__init__()
        self.block_idx, self.last_drop_p, self.C = block_idx, last_drop_p, embed_dim
        self.C, self.D = embed_dim, cond_dim
        self.use_crossattn = use_crossattn
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.attn = SelfAttention(block_idx=block_idx, embed_dim=embed_dim, num_heads=num_heads, attn_drop=attn_drop, proj_drop=drop, attn_l2_norm=attn_l2_norm, flash_if_available=flash_if_available, use_rope=use_rope)
        if self.use_crossattn:
            self.ca = CrossAttention(embed_dim=embed_dim, kv_dim=kv_dim, num_heads=num_heads, proj_drop=drop, cos_attn=cos_attn)
            self.ca_norm = norm_layer(embed_dim, elementwise_affine=True)

        self.ffn = FFN(in_features=embed_dim, hidden_features=round(embed_dim * mlp_ratio), drop=drop, fused_if_available=fused_if_available)

        self.ln_wo_grad = norm_layer(embed_dim, elementwise_affine=False)
        self.shared_aln = shared_aln
        if self.shared_aln:
            self.ada_gss = nn.Parameter(torch.randn(1, 1, 6, embed_dim) / embed_dim**0.5)
        else:
            lin = nn.Linear(cond_dim, 6*embed_dim)
            self.ada_lin = nn.Sequential(nn.SiLU(inplace=False), lin)

        self.fused_add_norm_fn = None

    # NOTE: attn_bias is None during inference because kv cache is enabled
    def forward(self, x, cond_BD, attn_bias, rope_batch=None, ca_kv=None, kv_update_indices=None, q_indices=None, return_kv=False):   # C: embed_dim, D: cond_dim
        if self.shared_aln:
            gamma1, gamma2, scale1, scale2, shift1, shift2 = (self.ada_gss + cond_BD).unbind(2) # 116C + B16C =unbind(2)=> 6 B1C
        else:
            gamma1, gamma2, scale1, scale2, shift1, shift2 = self.ada_lin(cond_BD).view(-1, 1, 6, self.C).unbind(2)

        x_sa = self.ln_wo_grad(x).mul(scale1.add(1)).add_(shift1)
        attn_result = self.attn(x_sa, attn_bias=attn_bias, rope_batch=rope_batch, kv_update_indices=kv_update_indices, q_indices=q_indices, return_kv=return_kv)
        if return_kv:
            x_sa, kv = attn_result
        else:
            x_sa = attn_result
            kv = None
        # >>> NEW: if we queried only a subset (tail), scatter back to full length for the residual add
        x_res = x_sa.mul(gamma1)                 # [B, Lq, C]

        # apply stochastic depth (drop path) on the small tensor (cheaper)
        x_res = self.drop_path(x_res)

        if (q_indices is not None) and (x_res.shape[1] != x.shape[1]):
            # in-place accumulate only at the updated positions (no big zero tensor)
            # q_indices: [B, Lq] (long)
            idx = q_indices.unsqueeze(-1).expand(-1, -1, x.shape[2])   # [B, Lq, C]
            x.scatter_add_(dim=1, index=idx, src=x_res)                # in-place residual update
        else:
            x = x + x_res                                                # full-length path

        x = x + self.ca(self.ca_norm(x), ca_kv).float()
        x = x + self.drop_path(self.ffn( self.ln_wo_grad(x).mul(scale2.add(1)).add_(shift2) ).mul(gamma2)) # this mul(gamma2) cannot be in-placed when FusedMLP is used
        if return_kv:
            return x, kv
        return x

    def forward_cached(self, x_tail, cond_BD, rope_batch, rope_start, prefix_kv, ca_kv):
        """
        Process only tail tokens using cached prefix K/V for self-attention.

        Args:
            x_tail:     [B, L_tail, C] hidden states for current-scale tokens
            cond_BD:    conditioning (shared_aln or ada_lin input)
            rope_batch: RoPE cache
            rope_start: int, RoPE offset = prefix_len
            prefix_kv:  (k_pfx, v_pfx) each [B, L_pfx, H, d] in BLHc
            ca_kv:      cross-attention KV tuple

        Returns:
            (x_tail [B, L_tail, C], (k_tail, v_tail) each [B, L_tail, H, d])
        """
        if self.shared_aln:
            gamma1, gamma2, scale1, scale2, shift1, shift2 = (self.ada_gss + cond_BD).unbind(2)
        else:
            gamma1, gamma2, scale1, scale2, shift1, shift2 = self.ada_lin(cond_BD).view(-1, 1, 6, self.C).unbind(2)

        x_sa = self.ln_wo_grad(x_tail).mul(scale1.add(1)).add_(shift1)
        x_sa, kv_tail = self.attn.forward_with_prefix_cache(
            x_sa, rope_batch=rope_batch, rope_start=rope_start, prefix_kv=prefix_kv
        )

        x_res = x_sa.mul(gamma1)
        x_res = self.drop_path(x_res)
        x_tail = x_tail + x_res

        x_tail = x_tail + self.ca(self.ca_norm(x_tail), ca_kv).float()
        x_tail = x_tail + self.drop_path(self.ffn(self.ln_wo_grad(x_tail).mul(scale2.add(1)).add_(shift2)).mul(gamma2))
        return x_tail, kv_tail

    def extra_repr(self) -> str:
        return f'shared_aln={self.shared_aln}'


class AdaLNBeforeHead(nn.Module):
    def __init__(self, C, D, norm_layer):   # C: embed_dim, D: cond_dim
        super().__init__()
        self.C, self.D = C, D
        self.ln_wo_grad = norm_layer(C, elementwise_affine=False)
        self.ada_lin = nn.Sequential(nn.SiLU(inplace=False), nn.Linear(D, 2*C))

    def forward(self, x_BLC: torch.Tensor, cond_BD: torch.Tensor):
        scale, shift = self.ada_lin(cond_BD).view(-1, 1, 2, self.C).unbind(2)
        return self.ln_wo_grad(x_BLC).mul(scale.add(1)).add_(shift)
