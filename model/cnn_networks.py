import torch
import torch.nn as nn
import torch.nn.init as init
from model.blocks import Resnet1D, SimpleConv1dLayer, Conv1dLayer


def length_to_mask(length, max_len=None, device: torch.device = None):
    if device is None:
        device = length.device

    if isinstance(length, list):
        length = torch.tensor(length)
    
    if max_len is None:
        max_len = max(length)

    length = length.to(device)
    mask = torch.arange(max_len, device=device).expand(
        len(length), max_len
    ).to(device) < length.unsqueeze(1)
    return mask

def init_weights(m):
    if isinstance(m, nn.Conv1d):
        # 使用 Xavier 初始化
        init.xavier_normal_(m.weight)
        if m.bias is not None:
            init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        # 使用 Xavier 初始化
        init.xavier_normal_(m.weight)
        if m.bias is not None:
            init.constant_(m.bias, 0)

def reparametrize(mu, logvar):
    s_var = logvar.mul(0.5).exp_()
    eps = s_var.data.new(s_var.size()).normal_()
    return eps.mul(s_var).add_(mu)

class GlobalRegressor(nn.Module):
    def __init__(self, dim_in, dim_latent, dim_out):
        super().__init__()
        layers = []
        layers.append(
            nn.Sequential(
                nn.Conv1d(dim_in, dim_latent, 3, 1, 1),
                nn.LeakyReLU(0.2)
                )
        )
        
        layers.append(Resnet1D(dim_latent, n_depth=3, dilation_growth_rate=3, reverse_dilation=True))
        layers.append(nn.Conv1d(dim_latent, dim_out, 3, 1, 1))
        self.layers = nn.Sequential(*layers)
        self.apply(init_weights)



    def forward(self, input):
        input = input.permute(0, 2, 1)
        return self.layers(input).permute(0, 2, 1)
    


############################################
################# VQ Model #################
############################################
class EncoderAttn(nn.Module):
    def __init__(self,
                 input_emb_width=3,
                 output_emb_width=512,
                 down_t=2,
                 stride_t=2,
                 width=512,
                 depth=3,
                 dilation_growth_rate=3,
                 activation='relu',
                 use_attn=False,
                 norm=None):
        super().__init__()

        filter_t, pad_t = stride_t * 2, stride_t // 2
        self.embed = nn.Sequential(
            nn.Conv1d(input_emb_width, width, 3, 1, 1),
            nn.ReLU()
        )

        self.res_blocks = nn.ModuleList()
        self.attn_blocks = nn.ModuleList()
        for i in range(down_t):
            input_dim = width
            block = nn.Sequential(
                nn.Conv1d(input_dim, width, filter_t, stride_t, pad_t),
                Resnet1D(width, depth, dilation_growth_rate, activation=activation, norm=norm),
            )
            self.res_blocks.append(block)
            self.attn_blocks.append(make_attn(width, use_attn=use_attn))
        self.outproj = nn.Conv1d(width, output_emb_width, 3, 1, 1)
        self.apply(init_weights)

    def forward(self, x, m_lens=None):
        x = self.embed(x)
        for res_block, attn_block in zip(self.res_blocks, self.attn_blocks):
            x = res_block(x)
            if m_lens is not None: m_lens = m_lens//2
            x = attn_block(x, m_lens)
        return self.outproj(x)


class DecoderAttn(nn.Module):
    def __init__(self,
                 input_emb_width=3,
                 output_emb_width=512,
                 down_t=2,
                 stride_t=2,
                 width=512,
                 depth=3,
                 dilation_growth_rate=3,
                 activation='relu',
                 use_attn = False,
                 norm=None):
        super().__init__()

        self.embed = nn.Sequential(
            nn.Conv1d(output_emb_width, width, 3, 1, 1),
            nn.ReLU()
        )

        self.res_blocks = nn.ModuleList()
        self.attn_blocks = nn.ModuleList()
        for i in range(down_t):
            out_dim = width
            block = nn.Sequential(
                Resnet1D(width, depth, dilation_growth_rate, reverse_dilation=True, activation=activation, norm=norm),
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv1d(width, out_dim, 3, 1, 1)
            )
            self.res_blocks.append(block)
            self.attn_blocks.append(make_attn(width, use_attn))

        self.outproj = nn.Sequential(
            nn.Conv1d(width, width, 3, 1, 1),
            nn.ReLU(),
            nn.Conv1d(width, input_emb_width, 3, 1, 1)
        )
        self.apply(init_weights)

    def forward(self, x, m_lens=None, keep_shape=False):
        x = self.embed(x)

        for res_block, attn_block in zip(self.res_blocks, self.attn_blocks):
            x = res_block(x)
            if m_lens is not None: m_lens *= 2
            x = attn_block(x, m_lens)

        x = self.outproj(x)

        if keep_shape:
            return x
        else:
            return x.permute(0, 2, 1)


def make_attn(in_channels, use_attn=True):
    return AttnBlock(in_channels) if use_attn else MultiInputIdentity()


class RotaryPositionEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE) for encoding relative positions."""
    def __init__(self, dim, max_seq_len=1024, base=10000):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base

        # Precompute inverse frequencies
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

        # Precompute cos/sin cache
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len):
        t = torch.arange(seq_len, device=self.inv_freq.device).float()
        freqs = torch.outer(t, self.inv_freq)  # (seq_len, dim/2)
        emb = torch.cat([freqs, freqs], dim=-1)  # (seq_len, dim)
        self.register_buffer('cos_cached', emb.cos(), persistent=False)
        self.register_buffer('sin_cached', emb.sin(), persistent=False)

    def forward(self, x, seq_len):
        """
        Args:
            x: (B, num_heads, T, head_dim)
            seq_len: sequence length
        Returns:
            x with rotary embeddings applied
        """
        if seq_len > self.max_seq_len:
            self._build_cache(seq_len)

        cos = self.cos_cached[:seq_len]  # (T, head_dim)
        sin = self.sin_cached[:seq_len]
        return self._apply_rotary(x, cos, sin)

    def _apply_rotary(self, x, cos, sin):
        # x: (B, num_heads, T, head_dim)
        # Rotate half the dimensions
        x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
        # Apply rotation
        rotated = torch.cat([-x2, x1], dim=-1)
        return x * cos + rotated * sin


class AttnBlock(nn.Module):
    """
    Enhanced attention block with:
    - Pre-norm self-attention (8 heads)
    - RoPE (Rotary Position Embedding) for temporal awareness
    - Optional gated attention (NeurIPS 2025)
    - FFN (Feed-Forward Network) after attention
    - Residual connections for both
    """
    def __init__(self, in_channels, num_heads=8, dropout=0.1, ffn_ratio=4, use_gate=True, use_rope=True, max_seq_len=256):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = in_channels // num_heads
        self.scale = self.head_dim ** -0.5
        self.use_gate = use_gate
        self.use_rope = use_rope

        # Pre-norm
        self.norm1 = nn.LayerNorm(in_channels)

        # Q, K, V projections (manual attention for RoPE support)
        self.q_proj = nn.Linear(in_channels, in_channels)
        self.k_proj = nn.Linear(in_channels, in_channels)
        self.v_proj = nn.Linear(in_channels, in_channels)
        self.out_proj = nn.Linear(in_channels, in_channels)

        # RoPE
        if use_rope:
            self.rope = RotaryPositionEmbedding(self.head_dim, max_seq_len=max_seq_len)

        # Dropout
        self.attn_dropout = nn.Dropout(dropout)

        # Gate projection: query-dependent, one gate per head
        if use_gate:
            self.gate_proj = nn.Linear(in_channels, num_heads)

        # FFN with pre-norm
        self.norm2 = nn.LayerNorm(in_channels)
        self.ffn = nn.Sequential(
            nn.Linear(in_channels, in_channels * ffn_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(in_channels * ffn_ratio, in_channels),
            nn.Dropout(dropout)
        )

    def forward(self, x, m_lens):
        # x: (B, C, T) -> (B, T, C)
        x = x.permute(0, 2, 1)
        B, T, C = x.shape

        # Create attention mask
        if m_lens is not None:
            key_mask = length_to_mask(m_lens, T, device=x.device)
            attn_mask = ~key_mask  # True = masked
        else:
            attn_mask = None

        # Pre-norm
        normed = self.norm1(x)

        # Project Q, K, V
        q = self.q_proj(normed).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(normed).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(normed).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE to Q and K
        if self.use_rope:
            q = self.rope(q, T)
            k = self.rope(k, T)

        # Compute attention scores
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # Apply mask
        if attn_mask is not None:
            attn_scores = attn_scores.masked_fill(attn_mask.unsqueeze(1).unsqueeze(2), float('-inf'))

        # Softmax + dropout
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        # Apply attention to values
        attn_out = torch.matmul(attn_weights, v)  # (B, num_heads, T, head_dim)

        # Apply gated attention
        if self.use_gate:
            gate_score = self.gate_proj(normed)  # (B, T, num_heads)
            gate_score = gate_score.transpose(1, 2).unsqueeze(-1)  # (B, num_heads, T, 1)
            attn_out = attn_out * torch.sigmoid(gate_score)

        # Reshape and project output
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, C)
        attn_out = self.out_proj(attn_out)

        x = x + attn_out

        # Pre-norm FFN + residual
        x = x + self.ffn(self.norm2(x))

        return x.permute(0, 2, 1)
    

class MultiInputIdentity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, m_lens=None):
        return x