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
    confidence_based_sampling: bool = True,  # whether to use confidence-based decoding
    alpha: float = 5.0,  # hyperparameter for window size calculation
    max_window: int = 32,  # Maximum window size for sliding window
    confidence_method: str = "prob_diff",  # "position", "top_prob", "prob_diff", "entropy"
    use_sliding_window: bool = False,  # whether to use sliding window for position selection
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

        # --- confidence-based decoding ---
        if confidence_based_sampling > 0.0:
            # Confidence-based unmasking (vectorized)
            mask_positions = (xt == mask)  # (B, L)
            num_mask_positions = mask_positions.sum(dim=1)  # (B,)
            
            # 1. Determine number of tokens to unmask using Poisson
            unmask_counts = torch.poisson(num_mask_positions.float() * dt).long()  # (B,)
            
            # 2. Calculate confidence based on selected method
            if confidence_method == "position":
                # Position-based confidence: position i / len(xt)
                xt_len = (xt != pad).sum(dim=1)  # (B,) - current sequence lengths
                position_indices = torch.arange(max_length, device=device).unsqueeze(0).expand(batch_size, -1)  # (B, L)
                confidence = 1.0 - (position_indices.float() / xt_len.unsqueeze(1).float().clamp(min=1))  # (B, L)
            
            elif confidence_method == "top_prob":
                # Top probability confidence
                import torch.nn.functional as F
                token_logits = unmask_rate  # (B, L, V) - use the unmask_rate as logits
                unmask_probs = F.softmax(token_logits, dim=-1)  # (B, L, V)
                confidence = unmask_probs.max(dim=-1)[0]  # (B, L)
            
            elif confidence_method == "prob_diff":
                # Probability difference confidence (top - second top)
                import torch.nn.functional as F
                token_logits = unmask_rate  # (B, L, V)
                unmask_probs = F.softmax(token_logits, dim=-1)  # (B, L, V)
                top2_probs, _ = torch.topk(unmask_probs, k=2, dim=-1)  # (B, L, 2)
                confidence = top2_probs[:, :, 0] - top2_probs[:, :, 1]  # (B, L)
            
            elif confidence_method == "entropy":
                # Entropy-based confidence (lower entropy = higher confidence)
                import torch.nn.functional as F
                token_logits = unmask_rate  # (B, L, V)
                unmask_probs = F.softmax(token_logits, dim=-1)  # (B, L, V)
                entropy = -torch.sum(unmask_probs * torch.log(unmask_probs + 1e-10), dim=-1)  # (B, L)
                confidence = -entropy  # (B, L) - negative entropy so lower entropy gives higher confidence
            
            else:
                raise ValueError(f"Unknown confidence_method: {confidence_method}")
            
            # 3. Apply window constraint if enabled
            if use_sliding_window:
                # Calculate dynamic k for each batch
                k_values = torch.minimum(
                    torch.minimum(
                        (alpha * unmask_counts).long(), 
                        torch.tensor(max_window, device=device)
                    ), num_mask_positions)  # (B,)
                
                # Get cumulative count of mask positions
                mask_cumsum = mask_positions.cumsum(dim=1)  # (B, L)
                
                # Create window mask: position is eligible if it's a mask and within first k masks
                is_within_window = mask_cumsum <= k_values.unsqueeze(1)  # (B, L)
                window_mask = mask_positions & is_within_window  # (B, L)
                
                # Set confidence to -inf for positions outside the window or non-mask positions
                confidence = torch.where(window_mask, confidence, torch.tensor(-float('inf'), device=device))
            else:
                # No window constraint - only mask positions are eligible
                confidence = torch.where(mask_positions, confidence, torch.tensor(-float('inf'), device=device))
            
            new_xt = xt.clone()

            # Vectorized unmasking
            max_unmask = unmask_counts.max().item()
            if max_unmask > 0:
                # Get top-k indices for all batches
                _, all_top_indices = torch.topk(confidence, k=max_unmask, dim=1, largest=True)  # (B, max_unmask)
                
                # Create mask for valid unmask operations
                unmask_mask = torch.arange(max_unmask, device=device).unsqueeze(0) < unmask_counts.unsqueeze(1)  # (B, max_unmask)
                
                # Get most likely tokens
                most_likely_tokens = unmask_rate.argmax(dim=-1)  # (B, L)
                
                # Gather the tokens to place at selected positions
                selected_positions = all_top_indices[unmask_mask]  # Flattened valid positions
                batch_indices = torch.arange(batch_size, device=device).unsqueeze(1).expand(-1, max_unmask)[unmask_mask]  # Corresponding batch indices
                
                # Apply unmasking with sampled tokens
                new_xt[batch_indices, selected_positions] = most_likely_tokens[batch_indices, selected_positions]
        else:
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
