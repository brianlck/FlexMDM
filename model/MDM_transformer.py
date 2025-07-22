import torch
import torch.nn.functional as F
from . import rotary
from .transformer import EmbeddingLayer, TimestepEmbedder, DDiTBlock, DDitFinalLayer
from omegaconf import OmegaConf
from torch.nn.attention.flex_attention import create_block_mask


def _dense_mask(b, h, q_idx, kv_idx):
    return torch.full_like(q_idx, True, dtype=torch.bool)


class DDiTNoLengthModel(torch.nn.Module):
    """
    A DDiT‐style model that predicts only per‐token posteriors,
    without any sequence‐length head, opt for the vanilla MDM
    """

    def __init__(self, config):
        super().__init__()
        # allowing dict configs too
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

        self.blocks = torch.nn.ModuleList(
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
        # final per‐token head only / no length head
        self.output_layer = DDitFinalLayer(
            config.model.hidden_size, self.vocab_size, config.model.cond_dim
        )

    def forward(self, indices: torch.Tensor, t: torch.Tensor):
        """
        indices: (B, L) token indices
        t:       (B,) timestep scalars
        returns: ReparametrizedRate with only per_token_posterior set
        """
        B, L = indices.shape

        block_mask = create_block_mask(
            _dense_mask, B=B, H=None, Q_LEN=indices.shape[1], KV_LEN=indices.shape[1]
        )
        print(block_mask)

        x = self.vocab_embed(indices)  # (B, L, hidden)
        c = F.silu(self.sigma_map(t))  # (B, cond_dim)
        rotary_cos_sin = self.rotary_emb(x)  # precompute rotary embeddings

        # run the stack
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            for i in range(len(self.blocks)):
                x = self.blocks[i](x, rotary_cos_sin, c, block_mask)

            token_logits = self.output_layer(x, c)
            return token_logits
