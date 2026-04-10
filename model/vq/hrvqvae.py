import random
import torch
import torch.nn as nn
from model.cnn_networks import EncoderAttn, DecoderAttn
from model.vq.quantizer import MSQuantizer

def length_to_mask(length, max_len, device: torch.device = None) -> torch.Tensor:
    if device is None:
        device = "cpu"

    if isinstance(length, list):
        length = torch.tensor(length)

    length = length.to(device)
    mask = torch.arange(max_len, device=device).expand(
        len(length), max_len
    ).to(device) < length.unsqueeze(1)
    return mask


class HRVQVAE(nn.Module):
    def __init__(self,
                 args,
                 input_width=263,
                 down_t=3,
                 stride_t=2,
                 width=512,
                 depth=3,
                 dilation_growth_rate=3,
                 activation='relu',
                 use_attn=False,
                 norm=None):

        super().__init__()
        output_emb_width = args.quantizer.code_dim
        self.encoder = EncoderAttn(input_width, output_emb_width, down_t, stride_t, width, depth,
                               dilation_growth_rate, activation=activation, norm=norm, use_attn=use_attn)
        self.decoder = DecoderAttn(input_width, output_emb_width, down_t, stride_t, width, depth,
                               dilation_growth_rate, activation=activation, norm=norm, use_attn=use_attn)
        self.cfg= args
        self.quantizer = MSQuantizer(nb_code=args.quantizer.nb_code, 
                                                code_dim=args.quantizer.code_dim, 
                                                mu=args.quantizer.mu, 
                                                scales=args.quantizer.scales,
                                                share_quant_resi=args.quantizer.share_quant_resi,
                                                quant_resi=args.quantizer.quant_resi)

        self.down_t = down_t

    def preprocess(self, x):
        # (bs, T, Jx3) -> (bs, Jx3, T)
        x = x.permute(0, 2, 1).float()
        return x

    def postprocess(self, x):
        # (bs, Jx3, T) ->  (bs, T, Jx3)
        x = x.permute(0, 2, 1)
        return x

    def encode_original(self, x, m_lens=None):
        x_in = self.preprocess(x)
        x_encoder = self.encoder(x_in, m_lens)   # UpTo196
        # Downsample m_lens to match encoder output temporal resolution
        m_lens_down = m_lens // (2 ** self.down_t) if m_lens is not None else None
        code_idx, all_codes, _ = self.quantizer.quantize_all(x_encoder, m_lens_down, return_latent=True)
        return code_idx, all_codes
    
    
    def encode(self, x, m_lens=None, perturb_rate=[0.0, 0.0], codebook=None, train=False):
        x_in = self.preprocess(x)
        x_encoder = self.encoder(x_in, m_lens)   # encoded result; encoder epsilon  [b, 512, 49]

        m_lens_down = m_lens // (2 ** self.down_t) if m_lens is not None else None
        
        code_idx, all_codes, accumulated_fhat = self.quantizer.quantize_all(x_encoder, m_lens_down, return_latent=True, perturb_rate=perturb_rate,
                                                            codebook=codebook, train=train)

        return code_idx, all_codes, accumulated_fhat

    def forward(self, x, m_lengths=None):
        x_in = self.preprocess(x)
        # Encode
        x_encoder = self.encoder(x_in, m_lengths)   # In train: [256, 263, 196]/UpTo196

        if m_lengths is not None:
            m_lengths //= 2**self.down_t
        ## quantization
        x_quantized, commit_loss, perplexity = self.quantizer(x_encoder, temperature=0.5, 
                                                              m_lens=m_lengths,
                                                              start_drop=self.cfg.quantizer.start_drop,
                                                              quantize_dropout_prob=self.cfg.quantizer.quantize_dropout_prob)  # maximum m_lengths is 49

        if m_lengths is not None:
            x_quantized = x_quantized.permute(0, 2, 1)
            mask = length_to_mask(m_lengths, x_quantized.shape[1])   #UpTo49
            x_quantized[~mask] = 0
            x_quantized = x_quantized.permute(0, 2, 1)   # UpTo49
        ## decoder
        x_out = self.decoder(x_quantized, m_lengths)
        return x_out, commit_loss, perplexity

    def forward_decoder(self, x, m_lengths=None):
        # m_lengths should be up to 196
        x_d = self.quantizer.get_codes_from_indices(x)
        if len(x_d.shape) == 4:
            raise NotImplementedError("Not implemented for multi-scale quantization yet.")
            x = x_d.sum(dim=0)

        if m_lengths is not None:
            m_lengths //= 2**self.down_t
            mask = length_to_mask(m_lengths, x_d.shape[1])
            x_d[~mask] = 0
        x_d = x_d.permute(0, 2, 1)

        # decoder
        x_out = self.decoder(x_d, m_lengths)
        return x_out

    def decode(self, x, m_lengths=None):
        # m_lengths should be up to 196

        if m_lengths is not None:
            x = x.permute(0, 2, 1)
            m_lengths //= 2**self.down_t
            mask = length_to_mask(m_lengths, x.shape[1], x.device)
            x[~mask] = 0
            x = x.permute(0, 2, 1)
        # decoder
        x_out = self.decoder(x, m_lengths)
        return x_out