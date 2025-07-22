import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange
from omegaconf import OmegaConf
from interpolant import ModelPrediction
from torch.nn.attention.flex_attention import flex_attention, create_block_mask
from . import rotary
from .fused_add_dropout_scale import (
    bias_dropout_add_scale_fused_train,
    bias_dropout_add_scale_fused_inference,
    modulate_fused,
)


flex_attention = torch.compile(flex_attention, mode="max-autotune")


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


#################################################################################
#                                  Layers                                       #
#################################################################################
class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones([dim]))
        self.dim = dim

    def forward(self, x):
        with torch.amp.autocast("cuda", enabled=False):
            x = F.layer_norm(x.float(), [self.dim])
        return x * self.weight[None, None, :]


#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size, frequency_embedding_size=256, silu=True):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32)
            / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """

    def __init__(self, num_classes, cond_size):
        super().__init__()
        self.embedding_table = nn.Embedding(num_classes + 1, cond_size)
        self.num_classes = num_classes

        # TODO think of initializing with 0.02 std deviation like in original DiT paper

    def forward(self, labels):
        embeddings = self.embedding_table(labels)
        return embeddings


# length scalar head
class ScalarLengthHead(nn.Module):
    def __init__(self, d_model: int, normalized_len: int, cond_dim: int | None = None):
        super().__init__()
        self.has_cond = cond_dim is not None
        if self.has_cond:
            self.adaLN = nn.Linear(cond_dim, 2 * d_model, bias=True)
            self.adaLN.weight.data.zero_()
            self.adaLN.bias.data.zero_()

        self.norm = LayerNorm(d_model)
        self.proj1 = nn.Linear(d_model, d_model)
        self.act = nn.GELU()
        self.proj2 = nn.Linear(d_model, 1)
        self.softplus = nn.Softplus()
        self.normalized_len = normalized_len

    def forward(self, x: torch.Tensor, c: torch.Tensor | None = None):
        x_fp32 = x.float()
        c_fp32 = c.float() if (self.has_cond and c is not None) else None
        if self.has_cond and c_fp32 is not None:
            shift, scale = self.adaLN(c_fp32)[:, None].chunk(2, dim=2)
            x_fp32 = modulate_fused(self.norm(x_fp32), shift, scale)
        else:
            x_fp32 = self.norm(x_fp32)
        s = self.proj2(self.act(self.proj1(x_fp32)))
        out = self.softplus(s).squeeze(-1) * self.normalized_len
        return out.to(x.dtype)


#################################################################################
#                                 Core Model                                    #
#################################################################################


def get_mask_mod(seq_len: torch.Tensor):
    def mask_mod(b, h, q_idx, kv_idx):
        return (q_idx <= seq_len[b]) & (kv_idx <= seq_len[b])

    return mask_mod


class DDiTBlock(nn.Module):
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
            nn.Linear(mlp_ratio * dim, dim, bias=True),
        )
        self.dropout2 = nn.Dropout(dropout)

        self.dropout = dropout

        self.adaLN_modulation = nn.Linear(cond_dim, 6 * dim, bias=True)
        self.adaLN_modulation.weight.data.zero_()
        self.adaLN_modulation.bias.data.zero_()

    def _get_bias_dropout_scale(self):
        return (
            bias_dropout_add_scale_fused_train
            if self.training
            else bias_dropout_add_scale_fused_inference
        )

    def forward(self, x, rotary_cos_sin, c, block_mask):
        batch_size = x.shape[0]

        bias_dropout_scale_fn = self._get_bias_dropout_scale()

        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c)[:, None].chunk(6, dim=2)
        )

        # attention operation
        x_skip = x
        x = modulate_fused(self.norm1(x), shift_msa, scale_msa)
        # dtype0 = x.dtype

        qkv = self.attn_qkv(x)
        qkv = rearrange(
            qkv, "b s (three h d) -> b s three h d", three=3, h=self.n_heads
        )
        with torch.amp.autocast("cuda", enabled=False):
            cos, sin = rotary_cos_sin
            qkv = rotary.apply_rotary_pos_emb(qkv, cos.to(qkv.dtype), sin.to(qkv.dtype))

        q, k, v = rearrange(qkv, "b s three h d -> three b h s d", three=3)

        x = flex_attention(q, k, v, block_mask=block_mask)

        x = rearrange(x, "b h s d -> b s (h d)", b=batch_size)

        x = bias_dropout_scale_fn(
            self.attn_out(x), None, gate_msa, x_skip, self.dropout
        )

        # mlp operation
        x = bias_dropout_scale_fn(
            self.mlp(modulate_fused(self.norm2(x), shift_mlp, scale_mlp)),
            None,
            gate_mlp,
            x,
            self.dropout,
        )

        return x


class EmbeddingLayer(nn.Module):
    def __init__(self, dim, vocab_dim):
        super().__init__()
        self.embedding = nn.Parameter(torch.empty((vocab_dim, dim)))
        torch.nn.init.kaiming_uniform_(self.embedding, a=math.sqrt(5))

    def forward(self, x):
        return self.embedding[x]


class DDitFinalLayer(nn.Module):
    def __init__(self, hidden_size, out_channels, cond_dim):
        super().__init__()
        self.norm_final = LayerNorm(hidden_size)
        self.linear = nn.Linear(hidden_size, out_channels)
        self.linear.weight.data.zero_()
        self.linear.bias.data.zero_()

        self.adaLN_modulation = nn.Linear(cond_dim, 2 * hidden_size, bias=True)
        self.adaLN_modulation.weight.data.zero_()
        self.adaLN_modulation.bias.data.zero_()

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c)[:, None].chunk(2, dim=2)
        x = modulate_fused(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class AnyOrderMaskInsertionFlow(nn.Module):
    def __init__(self, config):
        super().__init__()

        # hack to make loading in configs easier
        if isinstance(config, dict):
            config = OmegaConf.create(config)

        self.config = config
        self.vocab_size = config.interpolant.tokens
        self.pad_token = config.interpolant.pad_token
        self.mask_token = config.interpolant.mask_token

        self.vocab_embed = EmbeddingLayer(config.model.hidden_size, self.vocab_size)
        self.sigma_map = TimestepEmbedder(config.model.cond_dim)
        self.rotary_emb = rotary.Rotary(
            config.model.hidden_size // config.model.n_heads
        )

        self.blocks = nn.ModuleList(
            [
                DDiTBlock(
                    config.model.hidden_size,
                    config.model.n_heads,
                    config.model.cond_dim,
                    dropout=config.model.dropout,
                )
                for _ in range(config.model.n_blocks)
            ]
        )
        
        self.output_layer = DDitFinalLayer(
            config.model.hidden_size, self.vocab_size, config.model.cond_dim
        )

        self.len_predict_type = config.training.loss_fn.insert
        if self.len_predict_type == "distribution":
            self.len_pred = DDitFinalLayer(
                config.model.hidden_size,
                config.interpolant.max_length + 1,
                config.model.cond_dim,
            )
        elif self.len_predict_type == "expectation":
            normalized_len = config.interpolant.max_length
            self.len_pred = ScalarLengthHead(
                config.model.hidden_size, normalized_len, config.model.cond_dim
            )
        else:
            raise ValueError(f"Invalid length prediction type: {self.len_predict_type}")

    def _get_bias_dropout_scale(self):
        return (
            bias_dropout_add_scale_fused_train
            if self.training
            else bias_dropout_add_scale_fused_inference
        )

    def forward(self, indices: torch.Tensor, t: torch.Tensor):
        B, L = indices.shape
        indices = torch.cat(
            [
                indices,
                self.pad_token
                * torch.ones((B, 1), device=indices.device, dtype=torch.int64),
            ],
            dim=-1,
        )
        seq_lens = (indices != self.pad_token).sum(dim=-1)
        block_mask = create_block_mask(
            get_mask_mod(seq_lens),
            B=B,
            H=None,
            Q_LEN=indices.shape[1],
            KV_LEN=indices.shape[1],
        )

        x = self.vocab_embed(indices)
        c = F.silu(self.sigma_map(t))

        rotary_cos_sin = self.rotary_emb(x)

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            for i in range(len(self.blocks)):
                x = self.blocks[i](x, rotary_cos_sin, c, block_mask)

            # --- unmasking ---
            token_logits = self.output_layer(x[:, :-1], c)

            # --- length prediction ---
            match self.len_predict_type:
                case "distribution":
                    length_posterior = self.len_pred(x, c)
                    return ModelPrediction(
                        token_logits=token_logits,
                        length_posterior=length_posterior,
                    )
                case "expectation":
                    return ModelPrediction(
                        token_logits=token_logits,
                        expected_gaps=self.len_pred(x, c),
                    )
