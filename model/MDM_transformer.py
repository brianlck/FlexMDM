import torch
import torch.nn.functional as F
from . import rotary
from .transformer import (
    EmbeddingLayer,
    TimestepEmbedder,
    DDiTBlock,
    DDitFinalLayer
)
from interpolant import ModelPrediction

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

        self.vocab_embed = EmbeddingLayer(config.model.hidden_size, self.vocab_size + 1)
        self.sigma_map   = TimestepEmbedder(config.model.cond_dim)
        self.rotary_emb  = rotary.Rotary(config.model.hidden_size // config.model.n_heads)

        self.blocks = torch.nn.ModuleList([
            DDiTBlock(config.model.hidden_size, config.model.n_heads, config.model.cond_dim, dropout=config.model.dropout)
            for _ in range(config.model.n_blocks)
        ])
        # final per‐token head only / no length head
        self.output_layer = DDitFinalLayer(config.model.hidden_size, self.vocab_size, config.model.cond_dim)

    def forward(self, indices: torch.Tensor, t: torch.Tensor):
        """
        indices: (B, L) token indices
        t:       (B,) timestep scalars
        returns: ReparametrizedRate with only per_token_posterior set
        """
        B, L = indices.shape
        indices = torch.cat([indices, self.pad_token * torch.ones((B, 1), device=indices.device, dtype=torch.int64)], dim=-1)
        
        x = self.vocab_embed(indices)               # (B, L, hidden)
        c = F.silu(self.sigma_map(t))               # (B, cond_dim)
        rotary_cos_sin = self.rotary_emb(x)         # precompute rotary embeddings

        # run the stack
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            for i in range(len(self.blocks)):
                x = self.blocks[i](x, rotary_cos_sin, c, seqlens=None)

            per_token_posterior = self.output_layer(x[:, :-1], c)

        return ModelPrediction(
            token_posterior=per_token_posterior,
            length_posterior= None,
            variable_length=False
        )