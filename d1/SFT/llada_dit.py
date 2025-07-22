import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from llada_utils import load_llada_modules
(LLaDASequentialBlock, ModelConfig, ActivationType, LayerNormType) = load_llada_modules()

# ------------------------------------------------------------
# Additional scalar length head
# ------------------------------------------------------------
class ScalarLengthHead(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size)
        self.proj1 = nn.Linear(hidden_size, hidden_size)
        self.act = nn.GELU()
        self.proj2 = nn.Linear(hidden_size, 1)
        self.softplus = nn.Softplus()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        h = self.act(self.proj1(h))
        s = self.proj2(h)
        out = self.softplus(s).squeeze(-1)
        return out * 1024


class LLaDA_DIT(nn.Module):
    def __init__(self, backbone: nn.Module, pad_token_id: int, d_model: int):
        super().__init__()

        # define the architecture configuations
        self.backbone = backbone
        self.d_model = d_model
        self.pad_token_id = pad_token_id
        self._temb = None
        self.scalar_length_head = ScalarLengthHead(d_model)

        # define the time embedding
        self.time_mlp = nn.Sequential(
            nn.Linear(self.d_model, self.d_model * 4),
            nn.SiLU(),
            nn.Linear(self.d_model * 4, self.d_model),
        )

        # extract the core transformer blocks (backbone model is PEFT model)
        core = self.backbone.base_model.model.model.transformer

        # define the hook: change LN to AdaLN
        # ---- adding AdaLN for each block is costly, so we use a grouping AdaLN ----
        group_size = 8
        group_mod = None
        for i, block in enumerate(core.blocks):
            if i % group_size == 0:
                group_mod = nn.Linear(self.d_model, 2 * self.d_model)
                nn.init.zeros_(group_mod.weight) ; nn.init.zeros_(group_mod.bias)
                
            block.add_module("temb_mod", group_mod)

            for ln in (block.attn_norm, block.ff_norm):
                ln.register_forward_hook(self.make_hook(group_mod))

    def timestep_embedding(self, t, dim, max_period=10000):
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


    def make_hook(self, mod):
        # called before each LN forward
        def hook(module, _inp, output):
            if self._temb is None:
                return output
            scale, shift = mod(self._temb).chunk(2, dim=-1)
            x = (1 + scale[:, None, :]) * output + shift[:, None, :]
            return x
        return hook

    def forward(self, input_ids: torch.LongTensor, attention_mask: torch.Tensor | None = None, timesteps: torch.LongTensor = None, **backbone_kwargs):
        assert timesteps is not None, "timesteps must be provided"
        
        self._temb = self.time_mlp(
            self.timestep_embedding(timesteps, self.d_model)
        )
        input_ids = torch.cat(
            [input_ids, self.pad_token_id * torch.ones((input_ids.shape[0], 1), device = input_ids.device, dtype = torch.int64)]
            , dim = -1
        )
        out = self.backbone(input_ids = input_ids, attention_mask = attention_mask, output_hidden_states = True, return_dict = True, **backbone_kwargs)
        # get the unmasking prediction
        logits = out.logits[:, :-1, :]

        # get the length prediction
        hidden_v = out.hidden_states[-1] # extract the last hidden state
        length = self.scalar_length_head(hidden_v)

        # reset the time embedding for every forward pass
        self._temb = None

        return {"logits": logits, "length": length}
    
    @property
    def device(self):
        return self.backbone.device