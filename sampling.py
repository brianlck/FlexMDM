
# TOOD: This is a very inflexible sampling algorithm -- Only works for semiautoregressive with one token addition at one time
# TODO: This code is quite bad, we'd like to refactor, can we use ein / einops?
import torch
from dataclasses import dataclass
from interpolant import SemiAutoregressiveInterpolant
from typing import Any, Literal, Optional

@dataclass
class SamplingTraceDatapoint:
    t: float
    event_type: Literal["insertion", "change"] 
    position: int
    token: Any 

@dataclass
class SamplingResult:
    samples: torch.Tensor
    # Trace is supposed to be processed sequentially as updates are not commutative
    trace: Optional[list[SamplingTraceDatapoint]]

    def __iter__(self):
        yield from [self.samples, self.trace]
    
# Sample from categorical distribution for each position using the transition probabilities
def _sample_tokens(probs: torch.Tensor) -> torch.Tensor:
    """Sample one token per position from probability distribution.
    Args:
        probs: [batch_size, seq_len, vocab_size] transition probabilities
    Returns:
        [batch_size, seq_len] sampled token indices
    """
    batch_size, seq_len, vocab_size = probs.shape
    flat_probs = probs.view(-1, vocab_size)
    samples = torch.multinomial(flat_probs, num_samples=1)
    return samples.view(batch_size, seq_len) 

def semiauto_euler_sampling(
    model: torch.nn.Module,
    interpolant: SemiAutoregressiveInterpolant,
    steps: int,
    mask: int,
    pad: int,
    batch_size: int,
    max_length: int,
    return_trace: bool = False
) -> SamplingResult:
    device = model.device
    xt = pad * torch.ones((batch_size, max_length), device=device).to(torch.int64)
    sampling_trace = [[] for _ in range(batch_size)] if return_trace else None

    dt = 1.0 / steps
    t = torch.zeros((batch_size,), device=device)
    for i in range(steps):
        t = t + dt
        pred_rate = model(xt, t)
        pred_rate = interpolant.to_actual_rate(xt, pred_rate, t)
        unmask_rate = pred_rate.unmask_rate
        len_rate = pred_rate.length_rate
        # Probabilistically unmask token

        # Fix diagonal entries by setting mask positions to negative sum of other rates
        mask_positions = (xt == mask).nonzero(as_tuple=True)
        
        # Zero out rates for non-mask positions
        unmask_rate[xt != mask] = 0

        # First zero the mask token rate, so it's not included in the sum
        unmask_rate[*mask_positions,  mask] = 0
        # Then set it to negative sum of other rates
        unmask_rate[*mask_positions, mask] = -unmask_rate[*mask_positions, :].sum(dim=1)

        # Approximate probability with Euler step 1{x' = x} + Q dt
        trans_prob = unmask_rate * dt
        # Temporary xt variable that replaces pad with mask token to avoid illegal memory access
        _xt = xt.clone()
        _xt[xt == pad] = mask
        trans_prob.scatter_add_(2, _xt.unsqueeze(-1), torch.ones_like(_xt.unsqueeze(-1), dtype=unmask_rate.dtype))
        
        # Note that pad token is unchanged here
        new_xt = _sample_tokens(trans_prob.clamp(min=0, max=1))
        # Put pad back into palce
        new_xt[xt == pad] = pad

        # Sample if next token should be added
        first_pad_pos = torch.argmax((new_xt == pad).int(), dim=1)
        extension_prob = torch.clamp(len_rate * dt, min=0., max=1.)
        extension = torch.bernoulli(extension_prob).bool()
        # Find first pad position that will be replaced by mask
    
        # Replace pad token with mask
        extend_mask = first_pad_pos < max_length
        extend_mask = extend_mask & extension
        extend_id = extend_mask.nonzero()
        new_xt[extend_id, first_pad_pos[extend_id]] = mask

        new_xt = torch.where((xt != mask) & (xt != pad), xt, new_xt)

        if return_trace:
            for i in range(batch_size):
                
                # Check if the token was changed
                for j in range(max_length):
                    if xt[i, j] != pad and xt[i, j] != new_xt[i, j]:
                        sampling_trace[i].append(SamplingTraceDatapoint(
                            t=t[i].item(),
                            event_type="change",
                            position=j,
                            token=new_xt[i, j].item()
                        ))

                # Check if a new token was inserted
                if extend_mask[i]:
                    sampling_trace[i].append(SamplingTraceDatapoint(
                        t=t[i].item(),
                        event_type="insertion",
                        position=first_pad_pos[i].item(),
                        token=new_xt[i, first_pad_pos[i]].item()
                    ))
        
        xt = new_xt

    return SamplingResult(
        samples=xt,
        trace=sampling_trace
    )


def any_order_mask_insertion_euler_sampling(
    model: torch.nn.Module,
    interpolant: SemiAutoregressiveInterpolant,
    steps: int,
    mask: int,
    pad: int,
    batch_size: int,
    max_length: int,
    return_trace: bool = False,
) -> SamplingResult:
    device = model.device

    # 1) Initialize all‑pad sequence and trace
    xt = torch.full((batch_size, max_length), pad, dtype=torch.int64, device=device)
    sampling_trace = []

    dt = 1.0 / steps
    t = torch.zeros(batch_size, device=device)

    # Precompute row indices for scatter
    batch_idx_L    = torch.arange(batch_size, device=device).view(batch_size, 1).expand(batch_size, max_length)
    batch_idx_Lp1  = torch.arange(batch_size, device=device).view(batch_size, 1).expand(batch_size, max_length + 1)
    pos_idx_L      = torch.arange(max_length, device=device).view(1, max_length).expand(batch_size, max_length)
    gap_idx_Lp1    = torch.arange(max_length + 1, device=device).view(1, max_length + 1).expand(batch_size, max_length + 1)
    sampling_trace = [[] for _ in range(batch_size)] if return_trace else None

    for _ in range(steps):
        t = t + dt

        # ——— predict and convert rates ———
        pred_rate   = model(xt, t)
        pred_rate   = interpolant.to_actual_rate(xt, pred_rate, t)
        unmask_rate = pred_rate.unmask_rate    # (B, L, V)
        len_rate    = pred_rate.length_rate     # (B, L+1)

        # ——— unmask step (Euler) ———
        mask_pos = (xt == mask).nonzero(as_tuple=True)
        unmask_rate[xt != mask] = 0
        unmask_rate[*mask_pos, mask] = 0
        unmask_rate[*mask_pos, mask] = -unmask_rate[*mask_pos, :].sum(dim=1)
        trans_prob = (unmask_rate * dt).clamp(0.0, 1.0)

        # add “stay” probability
        _xt = xt.clone()
        _xt[xt == pad] = mask
        trans_prob.scatter_add_(2,
                                _xt.unsqueeze(-1),
                                torch.ones_like(_xt.unsqueeze(-1), dtype=trans_prob.dtype))

        new_xt = _sample_tokens(trans_prob)
        new_xt[xt == pad] = pad
        new_xt = torch.where((xt != mask) & (xt != pad), xt, new_xt)

        # ——— gap‑wise insertion, fully vectorized ———
        # 1) sample which gaps to extend
        ext       = torch.bernoulli((len_rate * dt).clamp(0.0, 1.0))  # (B, L+1)
        xt_len    = torch.sum(xt != pad, dim=1)  # (B,)
        ext       = torch.where(torch.arange(max_length + 1, device=device).view(1, max_length + 1) <= xt_len.view(batch_size, 1), ext, torch.zeros_like(ext))

        # 2) compute exclusive prefix sum of ext: number of inserts before each gap
        ext_ex    = ext.int().cumsum(dim=1)                                      # (B, L+1)

        # 3) compute new positions for every original token
        new_pos_orig = pos_idx_L + ext_ex[:, :max_length]                         # (B, L)
        valid_orig   = (new_pos_orig < max_length) & (pos_idx_L < xt_len.view(batch_size, 1))  # (B, L)
        # 4) compute new positions for every inserted mask
        new_pos_ins  = gap_idx_Lp1 + ext_ex - ext.int()                          # (B, L+1)
        valid_ins    = ext.bool() & (gap_idx_Lp1 <= xt_len.view(batch_size, 1))  # (B, L+1)

        # 5) build new tensor by scattering
        xt_tmp = torch.full_like(xt, pad)

        # scatter all tokens from new_xt into their shifted slots
        flat_b_o = batch_idx_L[valid_orig]
        flat_p_o = new_pos_orig[valid_orig]
        xt_tmp[flat_b_o, flat_p_o] = new_xt[valid_orig]

        # scatter new mask tokens at each gap insertion
        flat_b_i = batch_idx_Lp1[valid_ins]
        flat_p_i = new_pos_ins[valid_ins]
        xt_tmp[flat_b_i, flat_p_i] = mask

        if return_trace:
            # Check if the token was changed
            for i in range(batch_size):
                for j in range(max_length):
                    if xt[i, j] != pad and xt[i, j] != new_xt[i, j]:
                        sampling_trace[i].append(SamplingTraceDatapoint(
                            t=t[i].item(),
                            event_type="change",
                            position=j,
                            token=new_xt[i, j].item()
                        ))
                
                # Check if a new token was inserted
                for j in range(max_length):
                    id = max_length - j - 1
                    if ext[i, id]:
                        sampling_trace[i].append(SamplingTraceDatapoint(
                            t=t[i].item(),
                            event_type="insertion",
                            position=id,
                            token=mask
                        ))
        
        xt = xt_tmp

    return xt, sampling_trace