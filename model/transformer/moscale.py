import torch
import torch.nn as nn
import torch.nn.functional as F
from model.encode_text import T5TextEncoder
from functools import partial
from model.transformer.tools import *
from typing import Optional, Tuple
import math, random


from model.transformer.transformer_helper import AdaLNBeforeHead, AdaLNSelfAttn, FastRMSNorm, MultiQueryTextPool


def sample_with_top_k_top_p_(logits_BlV: torch.Tensor,
                            top_k: int = 0,
                            top_p: float = 0.0,
                            rng=None,
                            num_samples: int = 1,
                            temperature: float = 1.0) -> torch.Tensor:
    """
    Args:
        logits_BlV: [B, L, V] unnormalized logits
        top_k:     keep at most top_k tokens (0 = disabled)
        top_p:     nucleus threshold in (0,1]; keeps smallest set whose prob mass >= top_p (0/1 = disabled)
        rng:       torch.Generator for reproducible sampling
        num_samples: number of samples per position (>=1)
        temperature: >0. Scale logits before filtering & sampling

    Returns:
        indices [B, L, num_samples]
    """
    assert logits_BlV.dim() == 3, "Expected [B, L, V] logits"
    B, L, V = logits_BlV.shape
    assert num_samples >= 1, "num_samples must be >= 1"
    # Temperature scaling
    t = max(float(temperature), 1e-8)
    logits = logits_BlV / t

    # Start from: keep everything that isn't -inf
    keep = torch.isfinite(logits)

    # --- Top-k (cap strictly to k indices; avoids tie explosion) ---
    if top_k and 0 < top_k < V:
        topk_idx = logits.topk(top_k, dim=-1).indices                       # [B,L,top_k]
        keep_k = torch.zeros_like(logits, dtype=torch.bool).scatter(-1, topk_idx, True)
        keep = keep & keep_k

    # --- Top-p (nucleus) ---
    if top_p and 0.0 < top_p < 1.0:
        # Apply softmax only over currently-kept tokens
        masked_logits = logits.masked_fill(~keep, float('-inf'))            # [B,L,V]

        # Sort descending for standard nucleus filtering
        sorted_logits, sorted_idx = torch.sort(masked_logits, dim=-1, descending=True)  # [B,L,V]
        sorted_probs = torch.softmax(sorted_logits, dim=-1)                 # NaN-safe below
        cumsum = torch.cumsum(sorted_probs, dim=-1)

        # Remove tokens with cumprob > top_p; keep at least one token
        sorted_remove = cumsum > top_p
        sorted_remove[..., 0] = False

        # Map removal mask back to original vocab order
        remove_mask = torch.zeros_like(sorted_remove, dtype=torch.bool).scatter(-1, sorted_idx, sorted_remove)

        # Conjoin with existing keep
        keep = keep & (~remove_mask)

    # --- Guarantee at least one token kept per (B,L) ---
    any_kept = keep.any(dim=-1, keepdim=True)                               # [B,L,1]
    argmax_idx = logits.argmax(dim=-1, keepdim=True)                        # [B,L,1]
    force_one = torch.zeros_like(keep, dtype=torch.bool).scatter(-1, argmax_idx, True)
    keep = torch.where(any_kept, keep, force_one)

    # Final masked logits -> probabilities
    masked_logits = logits.masked_fill(~keep, float('-inf'))
    probs = torch.softmax(masked_logits, dim=-1)
    probs = torch.nan_to_num(probs, nan=0.0)
    probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)

    # Sample
    replacement = num_samples > 1
    idx = torch.multinomial(probs.view(-1, V), num_samples=num_samples, replacement=replacement, generator=rng)
    return idx.view(B, L, num_samples)


class SharedAdaLin(nn.Linear):
    def forward(self, cond_BD):
        C = self.weight.shape[0] // 6
        return super().forward(cond_BD).view(-1, 1, 6, C)   # B16C
    


@torch.no_grad()
def precompute_rope1d_for_batch(
    dim: int,
    lengths_per_sample,          # (N, 4) list/ndarray/tensor of ints: per-level valid lengths
    max_length: int,             # canonical length L*, e.g. 49
    pad_to_multiplier: int = 1,  # e.g., 128 for kernel-friendly shapes
    base: float = 10000.0,
    device=None,
    scaling_factor: float = 1.0,
):
    """
    Build scale-normalized 1D RoPE for a batch of variable per-level lengths.
    Outputs:
      rope_batch   : (N, 2, 1, 1, 1, L_batch_pad, dim//2)
      level_offsets: (N, 4) cumulative starts within each sample's concatenated sequence
      total_lens   : (N,)
    """
    assert dim % 2 == 0, "RoPE requires even head_dim per head"
    half = dim // 2

    # --- Master 1D cos/sin line on canonical grid [0..max_length-1] (fp32) ---
    inv_idx = torch.arange(0, half, 2, dtype=torch.float32, device=device) / half
    inv_freq = 1.0 / (base ** inv_idx)  # [half/2]

    t_line = torch.arange(max_length, dtype=torch.float32, device=device) / scaling_factor  # [L*]
    freqs = torch.outer(t_line, inv_freq)             # [L*, half/2]
    freqs = torch.repeat_interleave(freqs, 2, dim=-1) # [L*, half]  (even/odd share freq)
    master = torch.stack([torch.cos(freqs), torch.sin(freqs)], dim=0)  # [2, L*, half]

    # --- Normalize inputs ---
    lengths = torch.as_tensor(lengths_per_sample, device=device, dtype=torch.long)  # [N, 4]
    if lengths.ndim != 2 or lengths.size(1) != 4:
        raise ValueError("lengths_per_sample must be shape (N, 4)")
    N = lengths.size(0)

    # --- Build per-sample concatenated RoPE and offsets ---
    per_seq = []
    offsets = torch.zeros((N, 4), device=device, dtype=torch.long)
    total_lens = torch.zeros((N,), device=device, dtype=torch.long)

    for i in range(N):
        segs = []
        start = 0
        for lvl in range(4):
            L = int(lengths[i, lvl].item())
            offsets[i, lvl] = start
            if L > 0:
                # scale-normalized mapping: map each level position to canonical length
                idx = torch.round(
                    torch.arange(L, dtype=torch.float32, device=device) * (max_length / float(L))
                ).clamp_(0, max_length - 1).to(torch.long)  # [L]
                seg = master[:, idx, :]   # [2, L, half]
                segs.append(seg)
                start += L
            # if L == 0, just skip (no tokens at this level)
        total_lens[i] = start
        cat = torch.cat(segs, dim=1) if len(segs) else master.new_zeros((2, 0, half))
        per_seq.append(cat)  # [2, Li, half]

    # --- Batch pad to max total length (optionally rounded to multiplier) ---
    L_batch = int(total_lens.max().item()) if N > 0 else 0
    L_pad = L_batch

    # Allocate output and pack each sample's concatenated sequence
    rope_batch = master.new_zeros((N, 2, 1, 1, 1, L_pad, half))  # [N,2,1,1,1,L_pad,half]
    for i, cat in enumerate(per_seq):
        L_i = cat.shape[1]
        if L_i > 0:
            rope_batch[i, :, 0, 0, 0, :L_i, :] = cat  # fill valid tail; rest stays zero-padded

    return rope_batch, offsets, total_lens


############# Mask needed helpers
def cosine_schedule(t):
    return torch.cos(t * math.pi * 0.5)


def get_mask_subset_prob(mask, prob):
    subset_mask = torch.bernoulli(mask, p=prob) & mask
    return subset_mask


class MoScale(nn.Module):
    def __init__(self, code_dim, latent_dim=256,
                 num_heads=4, dropout=0.1, text_dim=512, cond_drop_prob=0.1, mlp_ratio=4,
                 device=None, cfg=None, full_length=80, scales=[8, 4, 2, 1],
                 shared_aln=False, norm_eps=1e-6, attn_drop_rate=0.,
                 flash_if_available=True, fused_if_available=True, token_embed_init=None,
                 rand_uncond=False):
        super(MoScale, self).__init__()

        self.code_dim = code_dim
        self.latent_dim = latent_dim
        self.text_dim = text_dim
        self.text_cond_length = 20
        self.cfg = cfg
        self.device = device
        self.full_length = full_length
        self.scales = scales
        self.patch_sizes = [int(full_length // scale) for scale in self.scales]
        self.L = sum(self.patch_sizes)
        self.first_l = self.patch_sizes[0]
        self.cond_drop_prob = cond_drop_prob
        self.num_heads = num_heads

        self.infer_use_kvcache = cfg.model.get('infer_use_kvcache', False)
        print(f'[MoScale] infer_use_kvcache = {self.infer_use_kvcache}')

        init_std = math.sqrt(1 / self.latent_dim / 3)

        self.num_layers = cfg.model.get('num_layers', 4)
        self.head_latent_dim = cfg.model.get('head_latent_dim', self.latent_dim)
        self.use_learned_token_map = cfg.model.get('use_learned_token_map', False)
        print(f'[MoScale] num_layers = {self.num_layers}')
        print(f'[MoScale] head_latent_dim = {self.head_latent_dim}')
        print(f'[MoScale] use_learned_token_map = {self.use_learned_token_map}')

        if self.use_learned_token_map:
            _num_tokens = cfg.vq.nb_code
            self.token_emb = nn.Parameter(torch.empty(_num_tokens, self.code_dim))
            nn.init.trunc_normal_(self.token_emb, mean=0, std=init_std)

        g = torch.Generator(device=self.device)
        g.manual_seed(cfg.exp.seed)
        self.rng = g

        self.perturb_rate = self.cfg.training.perturb_rate if ('perturb_rate' in self.cfg.training) else None
        print(f'[MoScale] perturb_rate = {self.perturb_rate}')

        self.input_process = nn.Linear(self.code_dim, self.latent_dim)

        # unconditional embedding for CFG
        cfg_uncond = torch.empty(self.text_cond_length, self.text_dim)
        rng = torch.Generator(device='cpu')
        rng.manual_seed(0)
        torch.nn.init.trunc_normal_(cfg_uncond, std=1.2, generator=rng)
        cfg_uncond /= self.text_dim ** 0.5
        self.cfg_uncond = nn.Parameter(cfg_uncond)


        # map the conditional features
        self.text_norm = FastRMSNorm(self.text_dim, elementwise_affine=True, eps=norm_eps)
        self.text_proj_for_sos = MultiQueryTextPool(self.text_dim, latent_dim, self.patch_sizes[0])
        # process the conditional features
        self.shared_ada_lin = nn.Sequential(nn.SiLU(inplace=False), SharedAdaLin(self.latent_dim, 6*self.latent_dim)) if shared_aln else nn.Identity()

        # backbone transformer blocks building
        norm_layer = partial(nn.LayerNorm, eps=norm_eps)
        self.use_crossattn = cfg.model.get('use_crossattn', False)
        if self.use_crossattn:
            self.text_proj_for_ca = nn.Sequential(
                nn.Linear(self.text_dim, latent_dim),
                nn.GELU(approximate='tanh'),
                nn.Linear(latent_dim, latent_dim),
                )

        ######################### prepare for MoScale #########################
        self.create_PE()

        attn_l2_norm = cfg.model.get('attn_l2_norm', True)
        print(f'[MoScale] attn_l2_norm = {attn_l2_norm}')

        num_layers = self.num_layers
        masked_drop_path_rate = 0.1 * num_layers/24
        masked_dpr = [x.item() for x in torch.linspace(0, masked_drop_path_rate, num_layers)]
        self.masked_blocks = nn.ModuleList([
            AdaLNSelfAttn(
                cond_dim=self.latent_dim, shared_aln=shared_aln,
                block_idx=block_idx, embed_dim=self.head_latent_dim, norm_layer=norm_layer, num_heads=num_heads, mlp_ratio=mlp_ratio,
                drop=dropout, attn_drop=attn_drop_rate, drop_path=masked_dpr[block_idx], last_drop_p=0 if block_idx == 0 else masked_dpr[block_idx-1],
                attn_l2_norm=attn_l2_norm,
                flash_if_available=flash_if_available, fused_if_available=fused_if_available,
                use_rope=self.mask_use_rope,
                use_crossattn=self.use_crossattn, kv_dim=self.latent_dim
            )
            for block_idx in range(num_layers)
        ])

        self.pad_id = -1
        self.mask_id = -2

        code_init_var = math.sqrt(1 / self.code_dim / 3)
        self.masked_token_embedding = nn.Parameter(torch.empty(1, self.code_dim))
        nn.init.trunc_normal_(self.masked_token_embedding.data, mean=0, std=code_init_var)
        self.padded_token_embedding = nn.Parameter(torch.empty(1, self.code_dim))
        nn.init.trunc_normal_(self.padded_token_embedding.data, mean=0, std=code_init_var)

        # project after concat: accumulated features + token embeddings -> block dim
        self.token_dim_proj = nn.Linear(self.code_dim, self.latent_dim)
        self.latent_dim_proj = nn.Linear(self.latent_dim*2, self.head_latent_dim)

        self.noise_schedule = cosine_schedule

        self.bert_replace_prob = 0.1
        self.bert_mask_prob = 0.8
        self.mask_augment_prob = [0.0, 0.1]

        self.sample_level_times = self.cfg.training.sample_level_times if ('sample_level_times' in self.cfg.training) else None
        print(f'[MoScale] sample_level_times = {self.sample_level_times}')
        assert len(self.patch_sizes) == len(self.sample_level_times)
        
        head_latent_dim = self.head_latent_dim

        # block-wise causal mask
        patch_sizes = self.patch_sizes
        L = sum(patch_sizes)
        d = torch.cat([torch.full((pn,), i, dtype=torch.long) for i, pn in enumerate(patch_sizes)]).view(1, L, 1)
        dT = d.transpose(1, 2)    # dT: 11L
        lvl_1L = dT[:, 0].contiguous()
        self.register_buffer('lvl_1L', lvl_1L)    # [1, 91] [[0*20,1,1,1,1,1,1,1,1,1,1,1,1,2...]]
        attn_bias_for_masking = torch.where(d >= dT, 0., -torch.inf).reshape(1, 1, L, L)
        self.register_buffer('attn_bias_for_masking', attn_bias_for_masking.contiguous())   #[1, 1, 91, 91] Block-wise causal mask


        # The head for the output
        self.vq_nb_code = cfg.vq.nb_code
        self.head_nm = AdaLNBeforeHead(head_latent_dim, self.latent_dim, norm_layer=norm_layer)
        self.head = nn.Linear(head_latent_dim, self.vq_nb_code)

        self.init_weights()

        self.text_emb = T5TextEncoder(
            device,
            local_files_only=False,
            from_pretrained=cfg.text_embedder.version,
            model_max_length=cfg.data.max_text_length
        )

        self.infer_rope_base = self.rope_base.repeat(64, 1, 1, 1, 1, 1, 1)


    def init_weights(self, init_adaln=0.5, init_adaln_gamma=1e-5, init_head=0.02, init_std=-1, conv_std_or_gain=0.02):
        if init_std < 0: init_std = (1 / self.latent_dim / 3) ** 0.5     # init_std < 0: automated
        
        print(f'[init_weights] {type(self).__name__} with {init_std=:g}')
        for m in self.modules():
            with_weight = hasattr(m, 'weight') and m.weight is not None
            with_bias = hasattr(m, 'bias') and m.bias is not None
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight.data, std=init_std)
                if with_bias: m.bias.data.zero_()
            elif isinstance(m, nn.Embedding):
                nn.init.trunc_normal_(m.weight.data, std=init_std)
                if m.padding_idx is not None: m.weight.data[m.padding_idx].zero_()
            elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm, nn.GroupNorm, nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d)):
                if with_weight: m.weight.data.fill_(1.)
                if with_bias: m.bias.data.zero_()
            # conv: VAR has no conv, only VQVAE has conv
            elif isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d)):
                if conv_std_or_gain > 0: nn.init.trunc_normal_(m.weight.data, std=conv_std_or_gain)
                else: nn.init.xavier_normal_(m.weight.data, gain=-conv_std_or_gain)
                if with_bias: m.bias.data.zero_()
        
        if init_head >= 0:
            if isinstance(self.head, nn.Linear):
                self.head.weight.data.mul_(init_head)
                self.head.bias.data.zero_()
            elif isinstance(self.head, nn.Sequential):
                self.head[-1].weight.data.mul_(init_head)
                self.head[-1].bias.data.zero_()
        
        if isinstance(self.head_nm, AdaLNBeforeHead):
            self.head_nm.ada_lin[-1].weight.data.mul_(init_adaln)
            if hasattr(self.head_nm.ada_lin[-1], 'bias') and self.head_nm.ada_lin[-1].bias is not None:
                self.head_nm.ada_lin[-1].bias.data.zero_()
        
        depth = len(self.masked_blocks)

        for block_idx, sab in enumerate(self.masked_blocks):
            # -------- scaling factor (Infinity-style) --------
            scale = math.sqrt(2 * depth)  # keep same policy as your original code (div_ by sqrt(2*depth))

            # -------- Self-Attn proj (supports .self_attn or .attn) --------
            sa = getattr(sab, "self_attn", None) or getattr(sab, "attn", None)
            if sa is not None and hasattr(sa, "proj") and hasattr(sa.proj, "weight"):
                sa.proj.weight.data.div_(scale)

            # -------- Cross-Attn proj (if the block has it) --------
            ca = getattr(sab, "cross_attn", None)
            if ca is not None and hasattr(ca, "proj") and hasattr(ca.proj, "weight"):
                ca.proj.weight.data.div_(scale)

            # -------- FFN proj (fc2) --------
            if hasattr(sab, "ffn") and hasattr(sab.ffn, "fc2") and hasattr(sab.ffn.fc2, "weight"):
                sab.ffn.fc2.weight.data.div_(scale)

            # optional FFN gating (if present)
            if hasattr(sab, "ffn") and getattr(sab.ffn, "fcg", None) is not None:
                nn.init.ones_(sab.ffn.fcg.bias)
                nn.init.trunc_normal_(sab.ffn.fcg.weight, std=1e-5)

            # -------- AdaLN parameters (6*C layout: [gamma1, gamma2, scale1, scale2, shift1, shift2]) --------
            # learned conditioner path
            if getattr(sab, "ada_lin", None) is not None:
                lin = sab.ada_lin[-1]  # final Linear that outputs 6*C
                if hasattr(lin, "weight"):
                    # first 2*C rows = gammas
                    lin.weight.data[: 2 * self.latent_dim].mul_(init_adaln_gamma)
                    # last 4*C rows = scale & shift
                    lin.weight.data[2 * self.latent_dim :].mul_(init_adaln)
                if hasattr(lin, "bias") and lin.bias is not None:
                    lin.bias.data.zero_()

            # shared conditioner path
            elif getattr(sab, "ada_gss", None) is not None:
                # sab.ada_gss: [1, 1, 6, C]
                sab.ada_gss.data[:, :, :2, :].mul_(init_adaln_gamma)  # gammas
                sab.ada_gss.data[:, :,  2:, :].mul_(init_adaln)       # scale/shift



    def create_PE(self):
        # Position Encoding: scale-normalized RoPE + lvl embedding + pos_start
        init_std = math.sqrt(1 / self.head_latent_dim / 3)
        num_heads = self.num_heads
        self.mask_use_rope = True
        self.lvl_embed = nn.Embedding(len(self.patch_sizes), self.head_latent_dim)
        nn.init.trunc_normal_(self.lvl_embed.weight.data, mean=0, std=init_std)
        self.rope_base, self.rope_offsets, self.rope_total_lens = precompute_rope1d_for_batch(self.head_latent_dim//num_heads, torch.tensor([self.patch_sizes]), max_length=self.patch_sizes[-1], device=self.device)
        self.pos_start = nn.Parameter(torch.empty(1, self.first_l, self.head_latent_dim))
        nn.init.trunc_normal_(self.pos_start.data, mean=0, std=init_std)

    def encode_text(self, raw_text):
        text_embedding, mask = self.text_emb.get_text_embeddings(raw_text)
        return text_embedding, mask

    def get_logits(self, h: torch.Tensor, cond_BD: Optional[torch.Tensor]):
        return self.head(self.head_nm(h.float(), cond_BD).float()).float()
    
    def get_next_autoregressive_input(self, si: int, SN: int, f_hat: torch.Tensor, h_BChw: torch.Tensor, non_pad_mask, vq_model) -> Tuple[Optional[torch.Tensor], torch.Tensor]: # only used in VAR inference
        feat_seq_len = self.patch_sizes[-1]
        current_mask = non_pad_mask[si].unsqueeze(-1).permute(0, 2, 1)  # (B, L, 1)
        h_BChw = h_BChw * current_mask.float()  # B, Cvae(512), L
        if si != SN-1:
            h = vq_model.quantizer.quant_resi[si/(SN-1)](F.interpolate(h_BChw, size=feat_seq_len, mode='linear'))     # conv after upsample
            f_hat.add_(h)
            return f_hat, F.interpolate(f_hat, size=self.patch_sizes[si+1], mode='area')
        else:
            h = vq_model.quantizer.quant_resi[si/(SN-1)](h_BChw)
            f_hat.add_(h)
            return f_hat, f_hat
    

    def preprocess_motion_for_training(self, motion, m_lens, vq_model, perturb_rate, train=False):
        vq_model.eval()

        id_list, raw_features, _ = vq_model.encode(motion[..., :self.cfg.data.dim_pose],
                                                m_lens.clone(), perturb_rate=perturb_rate, train=train)

        m_lens //= 2**vq_model.down_t

        non_pad_mask = []
        for scale, ele in zip(self.scales, id_list):
            ds_mlens = (m_lens // scale).long()
            non_pad_mask.append(lengths_to_mask(ds_mlens, ele.shape[1]))
        labels = torch.cat(id_list, dim=1)

        # downsample the raw_features to the size of the corresponding levels
        downsampled_feature_list = []
        for i in range(len(self.patch_sizes)-1):
            next_size = self.patch_sizes[i+1]
            downsampled_feature_list.append(F.interpolate(raw_features[i], size=next_size, mode='area'))

        x_BLC_wo_prefix = torch.cat(downsampled_feature_list, dim=2)

        return labels, x_BLC_wo_prefix.permute(0, 2, 1), torch.cat(non_pad_mask, dim=1), raw_features
    
    def dequantize(self, code_idx, vq_model):
        if not self.use_learned_token_map:
            return vq_model.quantizer.dequantize(code_idx)
        else:
            mask = code_idx == -1.
            code_idx = code_idx.masked_fill(mask, 0.)
            x = F.embedding(code_idx, self.token_emb)

            x[mask] = 0.
            return x

    def forward(self, motion, y, m_lens, vq_model, train=False):
        labels, x_BLC_wo_prefix, non_pad_mask, raw_features = self.preprocess_motion_for_training(motion, m_lens, vq_model, self.perturb_rate, train)
        B = x_BLC_wo_prefix.shape[0]

        with torch.no_grad():
            cond_embs, cond_att_mask = self.encode_text(y)
            cond_padding_mask = (cond_att_mask==0)

        max_seqlen_k = self.text_cond_length
        indi_length = [0]
        indi_feature = []

        for i in range(cond_embs.shape[0]):
            current_length = (~cond_padding_mask[i]).sum().item()
            indi_length.append(current_length)
            indi_feature.append(cond_embs[i, :current_length, :])

        cu_seqlens_k = torch.tensor(indi_length, device=self.device).cumsum(0).to(torch.int32)

        lens = indi_length[1:]
        kv_compact = torch.cat(indi_feature, dim=0)

        total = 0
        for le in lens:
            if random.random() < self.cond_drop_prob:
                kv_compact[total:total+le] = self.cfg_uncond[:le]
            total += le

        must_on_graph = self.cfg_uncond[0, 0] * 0   # trick for gradient graph in ddp
        kv_compact = self.text_norm(kv_compact).contiguous()
        sos = self.text_proj_for_sos((kv_compact, cu_seqlens_k, max_seqlen_k)).float().contiguous()
        cond_BD = sos.max(dim=1).values.unsqueeze(1)   # B, 1, C

        if self.use_crossattn:
            kv_compact = self.text_proj_for_ca(kv_compact).contiguous()
            kv_compact[0, 0] += must_on_graph
            ca_kv = kv_compact, cu_seqlens_k, max_seqlen_k
        else:
            ca_kv = None

        cond_BD_or_gss = self.shared_ada_lin(cond_BD).contiguous()

        # detect mixed precision dtype
        temp = x_BLC_wo_prefix.new_ones(8, 8)
        main_type = torch.matmul(temp, temp).dtype
        device = x_BLC_wo_prefix.device

        ids = labels.clone()
        padded_token = (ids == -1)
        non_pad_mask = ~padded_token

        ntokens = ids.shape[1]

        rand_time = uniform((B,), device=device)
        rand_mask_probs = self.noise_schedule(rand_time)
        num_token_masked = (ntokens * rand_mask_probs).round().clamp(min=1)

        batch_randperm = torch.rand((B, ntokens), device=device).argsort(dim=-1)
        mask = batch_randperm < num_token_masked.unsqueeze(-1)
        mask &= non_pad_mask

        labels = torch.where(mask, ids, self.mask_id)

        x_ids = ids.clone()
        # BERT-style masking: 10% replace, 80% mask, 10% keep
        mask_rid = get_mask_subset_prob(mask, self.bert_replace_prob)
        rand_id = torch.randint_like(x_ids, high=self.cfg.vq.nb_code)
        x_ids = torch.where(mask_rid, rand_id, x_ids)
        p_mid = self.bert_mask_prob / (1 - self.bert_replace_prob)
        mask_mid = get_mask_subset_prob(mask & ~mask_rid, p_mid)
        x_ids = torch.where(mask_mid, self.mask_id, x_ids)

        if self.mask_augment_prob is not None and self.mask_augment_prob[1] > 0:
            low, high = self.mask_augment_prob[0], self.mask_augment_prob[1]
            rand_id = torch.randint_like(x_ids, high=self.cfg.vq.nb_code)
            prob = random.uniform(low, high)
            mask_rand = (torch.rand(B, ntokens, device=device) < prob) & non_pad_mask & ~mask_mid
            x_ids = torch.where(mask_rand, rand_id, x_ids)

        input_token_embeddings = torch.zeros((B, ntokens, self.code_dim), device=device, dtype=main_type)

        non_pad_mask_position = (x_ids != self.pad_id) & (x_ids != self.mask_id)
        non_pad_mask_ids = x_ids[non_pad_mask_position]
        non_pad_mask_ids_token = self.dequantize(non_pad_mask_ids, vq_model)
        input_token_embeddings[non_pad_mask_position] = non_pad_mask_ids_token
        input_token_embeddings[x_ids == self.mask_id] = self.masked_token_embedding
        input_token_embeddings[x_ids == self.pad_id] = self.padded_token_embedding
        input_token_embeddings = self.token_dim_proj(input_token_embeddings)

        x_BLC_wo_prefix = self.input_process(x_BLC_wo_prefix)
        x_BLC = torch.cat((sos.expand(B, self.first_l, -1), x_BLC_wo_prefix), dim=1)

        attn_bias = self.attn_bias_for_masking.repeat(B, 1, 1, 1)

        x_BLC = x_BLC.to(dtype=main_type)
        cond_BD_or_gss = cond_BD_or_gss.to(dtype=main_type)
        attn_bias = attn_bias.to(dtype=main_type)

        x_BLC_token = torch.cat((x_BLC, input_token_embeddings), dim=-1)
        x_BLC_token = self.latent_dim_proj(x_BLC_token)

        for block_i, b in enumerate(self.masked_blocks):
            if block_i == 0:
                x_BLC_token[:, :self.first_l] = x_BLC_token[:, :self.first_l] + self.pos_start.expand(B, self.first_l, -1)
                x_BLC_token = x_BLC_token + self.lvl_embed(self.lvl_1L[:, :self.L].expand(B, -1))
            rope_batch = self.rope_base.repeat(B, 1, 1, 1, 1, 1, 1)
            x_BLC_token = b(x=x_BLC_token, cond_BD=cond_BD_or_gss, attn_bias=attn_bias, ca_kv=ca_kv, rope_batch=rope_batch)

        predict = self.get_logits(x_BLC_token.float(), cond_BD)

        # Loss/acc calculation
        labels_flatten = labels.reshape(-1)
        token_number = predict.shape[-1]
        pred_flatten = predict.reshape(-1, token_number)
        ce_loss = F.cross_entropy(pred_flatten, labels_flatten, ignore_index=self.mask_id)
        pred_id = pred_flatten.argmax(dim=1)
        mask = labels_flatten.ne(self.mask_id)
        acc = (pred_id == labels_flatten).masked_select(mask).float().mean().item()

        return ce_loss, pred_id, acc

    def extra_repr(self):
        return f'num_layers={self.num_layers}'

    @torch.no_grad()
    @eval_decorator
    def generate(self, conds, m_lens, cond_scale,
                 temperature=1, top_p_thres=0.9,
                 vq_model=None, sample_time=None):

        if sample_time is not None:
            self.sample_level_times = sample_time

        non_pad_mask = []
        for scale in self.scales:
            non_pad_mask.append(
                lengths_to_mask((m_lens//scale).long(), int(self.full_length//scale))
            )
        
        non_pad_mask_stack = torch.cat(non_pad_mask, dim=1).repeat(2, 1)  # [2*B, L]

        # first get the text conditions; FOR CFG (Classifier Freee Guidance)
        B = len(conds)
        cond_embs, cond_att_mask = self.encode_text(conds)
        cond_padding_mask = (cond_att_mask==0)

        # prepare the cond_embs
        max_seqlen_k = self.text_cond_length
        indi_length = [0]
        indi_feature = []

        for i in range(cond_embs.shape[0]):
            current_length = (~cond_padding_mask[i]).sum().item()
            indi_length.append(current_length)
            indi_feature.append(cond_embs[i, :current_length, :])

        cu_seqlens_k = torch.tensor(indi_length, device=self.device).cumsum(0).to(torch.int32)

        lens = indi_length[1:]
        kv_compact = torch.cat(indi_feature, dim=0)


        # prepare for the conds for CFG
        kv_compact_un = kv_compact.clone()
        total = 0
        for le in lens:
            kv_compact_un[total:total+le] = (self.cfg_uncond)[:le]
            total += le
        kv_compact = torch.cat((kv_compact, kv_compact_un), dim=0)
        cu_seqlens_k = torch.cat((cu_seqlens_k, cu_seqlens_k[1:]+cu_seqlens_k[-1]), dim=0)

        # begin the processing
        kv_compact = self.text_norm(kv_compact).contiguous()
        sos = self.text_proj_for_sos((kv_compact, cu_seqlens_k, max_seqlen_k)).float().contiguous()
        cond_BD = sos.max(dim=1).values.unsqueeze(1)   # B, 1, C

        if self.use_crossattn:
            kv_compact = self.text_proj_for_ca(kv_compact).contiguous()
            ca_kv = kv_compact, cu_seqlens_k, max_seqlen_k
        else:
            ca_kv = None


        cond_BD_or_gss = self.shared_ada_lin(cond_BD).contiguous()  # gss: gamma, scale, shift; cond_BD_or_gss should be float32

        feat_seq_len = self.patch_sizes[-1]; num_stages_minus_1 = len(self.patch_sizes) - 1
        f_hat = cond_BD.new_zeros(B, self.code_dim, feat_seq_len)
        return_list = []

        ######################## Per-scale inference with block-causal self-attn ########################
        attn_bias = self.attn_bias_for_masking.repeat(2*B, 1, 1, 1)

        next_token_map = sos.expand(2*B, self.patch_sizes[0], -1)
        cur_L = 0

        use_kvcache = self.infer_use_kvcache

        x_in = None
        prev_input_token_embeddings = None
        if use_kvcache:
            layer_kv_cache = [None] * len(self.masked_blocks)

        for i, pl in enumerate(self.patch_sizes):
            cur_L += pl

            if i == 0:
                x_in = next_token_map
            else:
                x_in = torch.cat([x_in, next_token_map], dim=1)
            x_out = x_in.clone()

            padding_mask = ~non_pad_mask_stack[:B, cur_L-pl:cur_L]

            cur_attn_bias = attn_bias[:, :, :cur_L, :cur_L]

            ############ Iterative remasking refinement ############
            cur_timestep = self.sample_level_times[i]
            ids = torch.where(padding_mask, self.pad_id, self.mask_id)
            scores = torch.where(padding_mask, 1e5, 0.)
            cur_length = pl

            for timestep_idx, timestep in enumerate(torch.linspace(0, 1, cur_timestep, device=x_out.device)):
                rand_mask_prob = self.noise_schedule(timestep)
                num_token_masked = torch.round(rand_mask_prob * cur_length).clamp(min=1)

                sorted_indices = scores.argsort(dim=1)
                ranks = sorted_indices.argsort(dim=1)
                is_mask = (ranks < num_token_masked.unsqueeze(-1))
                is_mask = is_mask & ~padding_mask
                ids = torch.where(is_mask, self.mask_id, ids)

                input_token_embeddings = torch.zeros((B, pl, self.code_dim), device=x_out.device, dtype=x_out.dtype)
                non_pad_mask_position = (ids != self.pad_id) & (ids != self.mask_id)
                input_token_embeddings[non_pad_mask_position] = self.dequantize(ids[non_pad_mask_position], vq_model)
                input_token_embeddings[ids == self.mask_id] = self.masked_token_embedding
                input_token_embeddings[ids == self.pad_id] = self.padded_token_embedding

                input_token_embeddings = self.token_dim_proj(input_token_embeddings)

                if prev_input_token_embeddings is not None:
                    input_token_embeddings = torch.concat([prev_input_token_embeddings, input_token_embeddings], dim=1)

                input_token_embeddings = input_token_embeddings.repeat(2, 1, 1)

                x_out_token = torch.cat((x_out, input_token_embeddings), dim=-1)
                x_out_token = self.latent_dim_proj(x_out_token)

                x_out_token[:, :self.first_l] = x_out_token[:, :self.first_l] + self.pos_start.expand(2*B, self.first_l, -1)
                x_out_token = x_out_token + self.lvl_embed(self.lvl_1L[:, :cur_L].expand(2*B, -1))
                rope_batch = self.infer_rope_base[:x_out_token.shape[0], ..., :cur_L, :]

                prefix_len = cur_L - pl

                if not use_kvcache:
                    for block_i, b in enumerate(self.masked_blocks):
                        x_out_token = b(x=x_out_token, cond_BD=cond_BD_or_gss, attn_bias=cur_attn_bias, ca_kv=ca_kv, rope_batch=rope_batch)
                elif timestep_idx == 0 or prefix_len == 0:
                    for block_i, b in enumerate(self.masked_blocks):
                        x_out_token, kv = b(x=x_out_token, cond_BD=cond_BD_or_gss,
                                            attn_bias=cur_attn_bias,
                                            ca_kv=ca_kv, rope_batch=rope_batch, return_kv=True)
                        if prefix_len > 0:
                            layer_kv_cache[block_i] = (kv[0][:, :prefix_len].clone(),
                                                       kv[1][:, :prefix_len].clone())
                        else:
                            layer_kv_cache[block_i] = None
                else:
                    x_tail = x_out_token[:, prefix_len:cur_L].clone()
                    for block_i, b in enumerate(self.masked_blocks):
                        x_tail, _ = b.forward_cached(x_tail, cond_BD_or_gss, rope_batch,
                                                     prefix_len, layer_kv_cache[block_i], ca_kv)
                    x_out_token = x_tail

                logits_BlV = self.get_logits(x_out_token.float(), cond_BD)
                t = cond_scale + (i-1) * (-0.25)
                logits_BlV = (1+t) * logits_BlV[:B] - t * logits_BlV[B:]
                logits_BlV = logits_BlV[:, -pl:, :]

                pred_ids = sample_with_top_k_top_p_(logits_BlV, rng=self.rng, top_k=0, top_p=top_p_thres, num_samples=1, temperature=temperature)[:, :, 0]
                ids = torch.where(is_mask, pred_ids, ids)

                probs_without_temperature = logits_BlV.softmax(dim=-1)
                scores = probs_without_temperature.gather(2, pred_ids.unsqueeze(dim=-1)).squeeze(-1)
                scores = scores.masked_fill(~is_mask, 1e5)

            ids = torch.where(padding_mask, self.pad_id, ids)
            idx_Bl = ids
            assert self.patch_sizes[i] == idx_Bl.shape[1]

            h_BChw = vq_model.quantizer.dequantize(idx_Bl).transpose(1, 2)
            h_BChw = h_BChw[:, :, -self.patch_sizes[i]:]

            cur_token_embeddings = self.dequantize(idx_Bl, vq_model).clone()
            padded_position = (idx_Bl == self.pad_id)
            cur_token_embeddings[padded_position] = self.padded_token_embedding
            if prev_input_token_embeddings is None:
                prev_input_token_embeddings = self.token_dim_proj(cur_token_embeddings)
            else:
                prev_input_token_embeddings = torch.cat([prev_input_token_embeddings, self.token_dim_proj(cur_token_embeddings)], dim=1)

            f_hat, next_token_map = self.get_next_autoregressive_input(i, len(self.patch_sizes), f_hat, h_BChw, non_pad_mask, vq_model)
            if i != num_stages_minus_1:
                next_token_map = next_token_map.transpose(1, 2)
                next_token_map = self.input_process(next_token_map)
                next_token_map = next_token_map.repeat(2, 1, 1)

            pred_idx = idx_Bl[..., -self.patch_sizes[i]:]
            pred_idx = torch.where(~non_pad_mask[i], -1, pred_idx)
            return_list.append(pred_idx)

        return return_list