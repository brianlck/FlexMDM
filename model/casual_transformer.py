import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from flash_attn.flash_attn_interface import flash_attn_varlen_qkvpacked_func
from .fused_add_dropout_scale import modulate_fused, bias_dropout_add_scale_fused_train, bias_dropout_add_scale_fused_inference
from .transformer import LayerNorm, EmbeddingLayer
from . import rotary


class CausalDiTBlock(nn.Module):
    def __init__(self, dim, n_heads, cond_dim, mlp_ratio=4, dropout=0.1):
        super().__init__()
        self.n_heads = n_heads
        self.norm1 = LayerNorm(dim)
        self.attn_qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.attn_out = nn.Linear(dim, dim, bias=False)
        self.dropout1 = nn.Dropout(dropout)
        self.norm2 = LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_ratio * dim, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_ratio * dim, dim, bias=True)
        )
        self.dropout2 = nn.Dropout(dropout)
        self.dropout = dropout
        # No time or label conditioning, so no adaLN_modulation

    def _get_bias_dropout_scale(self):
        return (
            bias_dropout_add_scale_fused_train
            if self.training
            else bias_dropout_add_scale_fused_inference
        )

    def forward(self, x, rotary_cos_sin, seqlens=None):
        batch_size, seq_len = x.shape[0], x.shape[1]
        bias_dropout_scale_fn = self._get_bias_dropout_scale()

        # attention operation
        x_skip = x
        x = self.norm1(x)
        # dtype0 = x.dtype

        qkv = self.attn_qkv(x)
        qkv = rearrange(qkv, 'b s (three h d) -> b s three h d', three=3, h=self.n_heads)
        with torch.cuda.amp.autocast(enabled=False):
            cos, sin = rotary_cos_sin
            qkv = rotary.apply_rotary_pos_emb(
                qkv, cos.to(qkv.dtype), sin.to(qkv.dtype)
            )
        qkv = rearrange(qkv, 'b s ... -> (b s) ...')
        if seqlens is None:
            cu_seqlens = torch.arange(
                0, (batch_size + 1) * seq_len, step=seq_len,
                dtype=torch.int32, device=qkv.device
            )
        else:
            cu_seqlens = seqlens.cumsum(-1)
        x = flash_attn_varlen_qkvpacked_func(
            qkv, cu_seqlens, seq_len, 0., causal=True)    
        x = rearrange(x, '(b s) h d -> b s (h d)', b=batch_size)
    
        scale = torch.ones(1, device=x.device, dtype=x.dtype)
        x = bias_dropout_scale_fn(self.attn_out(x), None, scale, x_skip, self.dropout)

        # mlp operation
        x = bias_dropout_scale_fn(
            self.mlp(self.norm2(x)), None, scale, x, self.dropout
        )
        return x

class CausalDiT(nn.Module):
    def __init__(self, config):
        super().__init__()
        if isinstance(config, dict):
            config = OmegaConf.create(config)

        self.config = config
        self.vocab_size = config.interpolant.tokens
        self.pad_token = config.interpolant.pad_token

        self.vocab_embed = EmbeddingLayer(config.model.hidden_size, self.vocab_size)
        self.rotary_emb = rotary.Rotary(config.model.hidden_size // config.model.n_heads)
        self.blocks = nn.ModuleList([
            CausalDiTBlock(config.model.hidden_size, config.model.n_heads, config.model.cond_dim, dropout=config.model.dropout)
            for _ in range(config.model.n_blocks)
        ])
        self.output_layer = nn.Linear(config.model.hidden_size, self.vocab_size)

    def forward(self, indices):
        x = self.vocab_embed(indices)
        rotary_cos_sin = self.rotary_emb(x)
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            for block in self.blocks:
                x = block(x, rotary_cos_sin, seqlens=None)
        logits = self.output_layer(x)
        return logits 