# TOOD: This is a very inflexible sampling algorithm -- Only works for semiautoregressive with one token addition at one time
# TODO: This code is quite bad, we'd like to refactor, can we use ein / einops?
import torch
from dataclasses import dataclass
from typing import Any, Literal, Optional

from lightning_modules.mdm import MaskedDiffusionModule


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


@torch.no_grad()
def mdm_euler_sampling(
    model: MaskedDiffusionModule,
    steps: int,
    mask: int,
    pad: int,
    batch_size: int,
    max_length: int,
    return_trace: bool = False,
):
    assert not return_trace, "Trace is not yet implemented in MDM Euler sampling"
    device = model.device
    xt = torch.full((batch_size, max_length), mask, dtype=torch.int64, device=device)

    dt = 1.0 / steps
    t = torch.zeros(batch_size, device=device)

    for i in range(steps):
        print("i-th sampling step")
        # ——— predict and convert rates ———
        pred_rate = model(xt, t)
        pred_rate = model.interpolant.to_actual_rate(xt, pred_rate, t)
        unmask_rate = pred_rate.unmask_rate

        # ——— unmask step (Euler) ———
        mask_pos = (xt == mask).nonzero(as_tuple=True)
        unmask_rate[xt != mask] = 0
        unmask_rate[*mask_pos, mask] = 0
        unmask_rate[*mask_pos, mask] = -unmask_rate[*mask_pos, :].sum(dim=1)
        trans_prob = (unmask_rate * dt).clamp(0.0, 1.0)

        _xt = xt.clone()
        trans_prob.scatter_add_(
            2,
            _xt.unsqueeze(-1),
            torch.ones_like(_xt.unsqueeze(-1), dtype=trans_prob.dtype),
        )

        if i == steps - 1:
            print("Final step, removing mask token from sampling")
            trans_prob[*mask_pos, mask] = 0.0
            print(trans_prob[*mask_pos, mask])

        new_xt = _sample_tokens(trans_prob)
        new_xt = torch.where(xt != mask, xt, new_xt)

        xt = new_xt
        t = t + dt

    return xt, []


@torch.no_grad()
def any_order_mask_insertion_euler_sampling(
    model: torch.nn.Module,
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
    batch_idx_L = (
        torch.arange(batch_size, device=device)
        .view(batch_size, 1)
        .expand(batch_size, max_length)
    )
    pos_idx_L = (
        torch.arange(max_length, device=device)
        .view(1, max_length)
        .expand(batch_size, max_length)
    )
    sampling_trace = [[] for _ in range(batch_size)] if return_trace else None

    for i in range(steps):
        # ——— predict and convert rates ———
        pred_rate = model(xt, t)
        pred_rate = model.interpolant.to_actual_rate(xt, pred_rate, t)
        unmask_rate = pred_rate.unmask_rate  # (B, L, V)
        len_rate = pred_rate.length_rate  # (B, L+1)

        # ——— unmask step (Euler) ———
        mask_pos = (xt == mask).nonzero(as_tuple=True)
        unmask_rate[xt != mask] = 0
        unmask_rate[*mask_pos, mask] = 0
        unmask_rate[*mask_pos, mask] = -unmask_rate[*mask_pos, :].sum(dim=1)
        trans_prob = (unmask_rate * dt).clamp(0.0, 1.0)

        # add “stay” probability
        _xt = xt.clone()
        _xt[xt == pad] = mask
        trans_prob.scatter_add_(
            2,
            _xt.unsqueeze(-1),
            torch.ones_like(_xt.unsqueeze(-1), dtype=trans_prob.dtype),
        )

        if i == steps - 1:
            print("Final step, removing mask token from sampling")
            trans_prob[*mask_pos, mask] = (
                0.0  # remove mask token from sampling at the last step
            )
            print(trans_prob[*mask_pos, mask])

        new_xt = _sample_tokens(trans_prob)
        new_xt[xt == pad] = pad
        new_xt = torch.where((xt != mask) & (xt != pad), xt, new_xt)

        if i != steps - 1:
            # ——— gap-wise insertion refactored — compute new length, fill masks, scatter tokens ———
            ext = torch.bernoulli((len_rate * dt).clamp(0.0, 1.0)).long()  # (B, L+1)
            xt_len = xt.ne(pad).sum(dim=1)  # (B,)
            gaps = torch.arange(max_length + 1, device=device).view(1, -1)
            ext = ext * (gaps <= xt_len.view(batch_size, 1)).long()
            total_ext = ext.sum(dim=1)
            valid = xt_len + total_ext <= max_length
            ext = ext * valid.view(batch_size, 1).long()

            ext_ex = ext.int().cumsum(dim=1)  # (B, L+1)
            new_len = xt_len + total_ext  # (B,)

            xt_tmp = torch.full_like(xt, pad)
            mask_fill = pos_idx_L < new_len.view(batch_size, 1)
            xt_tmp[mask_fill] = mask

            new_pos_orig = pos_idx_L + ext_ex[:, :max_length]  # (B, L)
            orig_mask = pos_idx_L < xt_len.view(batch_size, 1)
            flat_b = batch_idx_L[orig_mask]
            flat_p = new_pos_orig[orig_mask]
            xt_tmp[flat_b, flat_p] = new_xt[orig_mask]
        else:
            xt_tmp = new_xt

        if return_trace:
            # Check if the token was changed
            for i in range(batch_size):
                for j in range(max_length):
                    if xt[i, j] != pad and xt[i, j] != new_xt[i, j]:
                        sampling_trace[i].append(
                            SamplingTraceDatapoint(
                                t=t[i].item(),
                                event_type="change",
                                position=j,
                                token=new_xt[i, j].item(),
                            )
                        )

                # Check if a new token was inserted
                for j in range(max_length):
                    id = max_length - j - 1
                    if ext[i, id]:
                        sampling_trace[i].append(
                            SamplingTraceDatapoint(
                                t=t[i].item(),
                                event_type="insertion",
                                position=id,
                                token=mask,
                            )
                        )

        xt = xt_tmp
        t = t + dt

    return xt, sampling_trace


@torch.no_grad()
def mdm_tau_leaping_sampling(
    model: MaskedDiffusionModule,
    steps: int,
    mask: int,
    pad: int,
    batch_size: int,
    max_length: int,
    return_trace: bool = False,
):
    assert not return_trace, "Trace is not yet supported"
    device = model.device
    xt = torch.full((batch_size, max_length), mask, dtype=torch.int64, device=device)
    dt = 1.0 / steps
    t = torch.zeros(batch_size, device=device)

    for i in range(steps):
        # ——— predict and convert rates ———
        pred = model(xt, t)
        pred = model.interpolant.to_actual_rate(xt, pred, t)
        unmask_rate = pred.unmask_rate  # (B, L, V)

        if i == steps - 1:
            # last step: deterministic unmask via argmax
            mask_pos = xt == mask  # (B, L)
            new_token = unmask_rate.argmax(dim=2)  # (B, L)
            new_xt = xt.clone()
            new_xt[mask_pos] = new_token[mask_pos]
            new_xt = torch.where(xt != mask, xt, new_xt)
            xt = new_xt
            t = t + dt
            continue
        # tau-leaping via Poisson counts
        counts = torch.poisson(unmask_rate * dt).long()
        mask_pos = xt == mask  # (B, L)
        # zero out non-mask positions and mask→mask
        counts[~mask_pos.unsqueeze(-1).expand_as(counts)] = 0
        counts[..., mask] = 0
        # only accept exactly one event
        sum_c = counts.sum(dim=2)  # (B, L)
        one_event = sum_c == 1
        new_token = counts.argmax(dim=2)  # (B, L)

        # build new xt
        new_xt = xt.clone()
        new_xt[one_event] = new_token[one_event]
        # keep pads and already-unmasked tokens
        new_xt = torch.where(xt != mask, xt, new_xt)
        xt = new_xt
        t = t + dt

    return xt, []

# Not used in production, for debugging purposes
lengths = {4: 0.1, 16: 0.4, 32: 0.4, 64: 0.1}

def binomial_mass(k, n, p):
    """
    Calculate the probability mass function (PMF) for a binomial distribution.
    
    Args:
        k (int): Number of successes
        n (int): Number of trials
        p (float): Probability of success in a single trial
        
    Returns:
        float: Probability mass P(X = k)
    """
    import math
    
    # Calculate binomial coefficient (n choose k)
    try:
        binom_coef = math.factorial(n) / (math.factorial(k) * math.factorial(n - k))
    except ValueError:
        # Handle cases where k > n or negative values
        return 0.0
        
    # Calculate probability mass
    return binom_coef * (p ** k) * ((1 - p) ** (n - k))

def calculate_rate_batch(alpha_t, len_t):
    """
    Calculate rate for a batch of alpha_t and len_t values.
    
    Args:
        alpha_t (torch.Tensor): Tensor of shape (batch_size,)
        len_t (torch.Tensor): Tensor of shape (batch_size,)
        
    Returns:
        torch.Tensor: Tensor of shape (batch_size,) containing calculated rates
    """
    batch_size = alpha_t.shape[0]
    device = alpha_t.device
    
    # Initialize tensors for numerator and denominator
    nom = torch.zeros(batch_size, device=device)
    denom = torch.zeros(batch_size, device=device)
    
    for length, probability in lengths.items():
        # Create mask for valid entries where len_t <= length
        valid_mask = (len_t <= length) & (len_t >= 0)
        
        if not valid_mask.any():
            continue
        
        valid_indices = valid_mask.nonzero(as_tuple=True)[0]
        valid_len_t = len_t[valid_indices]
        valid_alpha_t = alpha_t[valid_indices]
        
        # Calculate binomial probabilities efficiently using torch distribution
        binom_dist = torch.distributions.Binomial(total_count=length, probs=valid_alpha_t)
        binom_probs = binom_dist.log_prob(valid_len_t).exp()
        
        # Update numerator and denominator for valid indices
        nom[valid_indices] += (length - valid_len_t) * probability * binom_probs
        denom[valid_indices] += probability * binom_probs
    
    # Handle division by zero in a vectorized way
    result = torch.zeros_like(nom)
    div_mask = denom > 0
    result[div_mask] = nom[div_mask] / (denom[div_mask])
    
    return result

# Keep the original function for backward compatibility
def calculate_rate(alpha_t, len_t):
    """Legacy scalar version of calculate_rate"""
    if isinstance(alpha_t, torch.Tensor) and alpha_t.ndim > 0:
        return calculate_rate_batch(alpha_t, len_t)
    
    nom, denom = 0, 0
    for length, probability in lengths.items():
        if length >= len_t:
            nom += (length - len_t) * probability * binomial_mass(len_t, length, alpha_t)
            denom += probability * binomial_mass(len_t, length, alpha_t)
    
    if denom == 0:
        return 0.0
    
    return nom /denom


@torch.no_grad()
@torch.compile(mode="reduce-overhead")
def any_order_mask_insertion_tau_leaping_sampling(
    model: torch.nn.Module,
    steps: int,
    mask: int,
    pad: int,
    batch_size: int,
    max_length: int,
    return_trace: bool = False,
) -> SamplingResult:
    device = model.device
    xt = torch.full((batch_size, max_length), pad, dtype=torch.int64, device=device)
    sampling_trace = []
    dt = 1.0 / steps
    t = torch.zeros(batch_size, device=device)

    # Precompute row indices for scatter
    batch_idx_L = (
        torch.arange(batch_size, device=device)
        .view(batch_size, 1)
        .expand(batch_size, max_length)
    )
    pos_idx_L = (
        torch.arange(max_length, device=device)
        .view(1, max_length)
        .expand(batch_size, max_length)
    )

    for i in range(steps):
        # --- predict rates ---
        pred = model(xt, t)
        xt_len = (xt != pad).sum(dim=1)
        pred = model.interpolant.to_actual_rate(xt, pred, t)
        unmask_rate = pred.unmask_rate  # (B, L, V)
        len_rate = pred.length_rate  # (B, L+1)

        if i == steps - 1:
            # last step: deterministic unmask via argmax
            mask_pos = xt == mask
            new_token = unmask_rate.argmax(dim=2)
            new_xt = xt.clone()
            new_xt[mask_pos] = new_token[mask_pos]
            new_xt = torch.where(xt == pad, pad, new_xt)
            new_xt = torch.where((xt != mask) & (xt != pad), xt, new_xt)
            xt = new_xt
            t = t + dt
            continue
        # --- tau-leaping unmask via Poisson ---
        counts = torch.poisson(unmask_rate * dt).long()
        mask_pos = xt == mask
        counts[~mask_pos.unsqueeze(-1).expand_as(counts)] = 0
        counts[..., mask] = 0
        sum_c = counts.sum(dim=2)
        one_event = sum_c == 1
        new_token = counts.argmax(dim=2)
        new_xt = xt.clone()
        new_xt[one_event] = new_token[one_event]
        new_xt = torch.where(xt == pad, pad, new_xt)
        new_xt = torch.where((xt != mask) & (xt != pad), xt, new_xt)

        # insertion only on non-last
        if i != steps - 1:
            # --- Poisson insertion, compute new lengths and fill masks ---
            ext = torch.poisson(len_rate * dt).long()  # (B, L+1)
            xt_len = xt.ne(pad).sum(dim=1)  # (B,)
            gaps = torch.arange(max_length + 1, device=device).view(1, -1)
            ext = ext * (gaps <= xt_len.view(batch_size, 1)).long()
            total_ext = ext.sum(dim=1)
            valid = xt_len + total_ext <= max_length
            ext = ext * valid.view(batch_size, 1).long()

            # compute prefix sums of insertions
            ext_ex = ext.int().cumsum(dim=1)  # (B, L+1)
            new_len = xt_len + total_ext  # (B,)

            # initialize with pads, then fill mask up to new_len
            xt_tmp = torch.full_like(xt, pad)
            mask_pos = pos_idx_L < new_len.view(batch_size, 1)
            xt_tmp[mask_pos] = mask

            # shift and scatter original tokens
            new_pos_orig = pos_idx_L + ext_ex[:, :max_length]  # (B, L)
            orig_mask = pos_idx_L < xt_len.view(batch_size, 1)
            flat_b = batch_idx_L[orig_mask]
            flat_p = new_pos_orig[orig_mask]
            xt_tmp[flat_b, flat_p] = new_xt[orig_mask]
        else:
            xt_tmp = new_xt

        xt = xt_tmp
        t = t + dt
        if return_trace:
            sampling_trace.append(xt)

    return xt, sampling_trace
