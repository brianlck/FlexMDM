import torch
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
import torch.distributed as dist
from generate import add_gumbel_noise
from typing import Optional, List, Tuple
import matplotlib.pyplot as plt
from sft_interpolant import ModelPrediction



def clean_decode(ids, *, mask_id, tokenizer):
    """
    Convert a list/1-D tensor of token-ids to a human-readable string.
    - strips EOS
    - keeps MASK (re-writes it to literal "[MASK]" so it's obvious)
    """
    eos = tokenizer.eos_token_id
    tokens = tokenizer.convert_ids_to_tokens(ids)

    pretty = []
    for tid, t in zip(ids, tokens):
        if tid == eos:              # skip EOS completely
            continue
        if tid == mask_id:          # show mask explicitly
            pretty.append("[MASK]")
        else:
            pretty.append(t)

    return tokenizer.convert_tokens_to_string(pretty).strip()

def unmasking_rate_calc(s: float, t: float, beta: float) -> float:
    delta = s - t
    denominator = (1-t) - beta * ((1-t)**2)/2
    numerator = delta - beta * (delta**2)/2
    return numerator / denominator if s < 1 - 1e-6 else 1.0

def insertion_rate_calc(s: float, t: float, beta: float) -> float:
    if beta == 0.0:
        return 0.0
    else:
        denominator = (1-t)**2
        numerator = 2 * (s-t) * (1-s)
        return numerator / denominator if s < 1 - 1e-6 else 0.0

def valid_insertion_positions(xt: torch.Tensor, pad: int, prompt_len: int) -> torch.Tensor:
    not_pad_tokens = (xt != pad) # B X L
    left_pad_tensor = torch.ones((xt.shape[0], 1), dtype = torch.bool, device = xt.device)
    valid_positions = torch.cat([left_pad_tensor, not_pad_tokens], dim = 1) # B X (L+1)
    valid_positions[:, : prompt_len] = False
    return valid_positions

def length(xt: torch.Tensor, pad: int, mask: int) -> torch.Tensor:
    seq_len = (xt != pad).sum(dim = 1)
    clean_tokens = ((xt != mask) & (xt != pad)).sum(dim = 1)
    return seq_len, clean_tokens

def plot_len_trace(len_trace: List[Tuple[int, int, int]], model_name: str):
    plt.figure(figsize=(10, 5))
    steps, seq_lens, clean_tokens = zip(*len_trace)
    plt.plot(steps, seq_lens, label = "Sequence length")
    plt.plot(steps, clean_tokens, label = "Number of clean tokens")
    plt.xlabel("Generation step")
    plt.ylabel("Number")
    plt.title("Quantity changes over generation steps")
    plt.legend()
    plt.savefig(f"len_trace_{model_name}.png")
    plt.close()


from sft_interpolant import AnyOrderMaskInsertionInterpolant, GeometricSchedule, LinearSchedule

@torch.no_grad()
def any_order_mask_insertion_tau_leaping_sampling(
    model: torch.nn.Module,
    steps: int,
    mask: int,
    pad: int,
    batch_size: int,
    max_length: int,
    prompts: Optional[torch.Tensor] = None,
    confidence_based_sampling: bool = True,
    return_trace: bool = False,
    alpha: float = 5.0,  # hyperparameter for window size calculation
    confidence_method: str = "prob_diff",  # "position", "top_prob", "prob_diff", "entropy"
    use_sliding_window: bool = True,  # whether to use sliding window for position selection
):
    insertion_schedule = GeometricSchedule(min_val=10.0, max_val=0.01)
    unmasking_schedule = LinearSchedule()

    interpolant = AnyOrderMaskInsertionInterpolant(
        insertion_schedule=insertion_schedule,
        unmask_schedule=unmasking_schedule,
        vocab_size=0,
        mask_token = mask,
        pad_token = pad,
        max_length = 1024
    )

    device = model.device
    
    # Initialize with prompts if provided, otherwise all pad tokens
    if prompts is not None:
        if prompts.size(0) == 1 and batch_size > 1:
            prompts = prompts.expand(batch_size, -1)
        
        # Vectorized prompt handling
        if prompts.size(1) > max_length:
            prompts = prompts[:, :max_length]
        
        xt = torch.full((batch_size, max_length), pad, dtype=torch.int64, device=device)
        
        # Get individual prompt lengths for each batch element
        prompt_lens = (prompts != pad).sum(dim=1)  # (B,) - individual lengths per batch
        
        # Create vectorized mask for copying prompts
        pos_indices = torch.arange(max_length, device=device).unsqueeze(0)  # (1, L)
        prompt_mask = pos_indices < prompt_lens.unsqueeze(1)  # (B, L)
        xt[prompt_mask] = prompts[prompt_mask]
        
        # Create vectorized prompt length mask for insertion protection
        prompt_len_mask = pos_indices < prompt_lens.unsqueeze(1)  # (B, L) - for protecting positions
    else:
        xt = torch.full((batch_size, max_length), pad, dtype=torch.int64, device=device)
        prompt_lens = torch.zeros(batch_size, dtype=torch.long, device=device)
        prompt_len_mask = torch.zeros((batch_size, max_length), dtype=torch.bool, device=device)
    
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
        pred = model(input_ids = xt, timesteps = t)
        pred = ModelPrediction(
            token_logits=pred["logits"],
            expected_gaps=pred["length"],
        )
        xt_len = (xt != pad).sum(dim=1)
        pred_rate = interpolant.to_actual_rate(xt, pred, t)
        unmask_rate = pred_rate.unmask_rate  # (B, L, V)
        len_rate = pred_rate.length_rate  # (B, L+1)

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

        if confidence_based_sampling:
            # Confidence-based unmasking (vectorized)
            mask_positions = (xt == mask)  # (B, L)
            num_mask_positions = mask_positions.sum(dim=1)  # (B,)
            
            # 1. Determine number of tokens to unmask using Poisson
            rate_scale_factor = unmasking_schedule.rate_scale_factor(t)
            unmask_counts = torch.poisson(num_mask_positions.float() * rate_scale_factor * dt).long()  # (B,)
            
            # 2. Calculate confidence based on selected method
            if confidence_method == "position":
                # Position-based confidence: position i / len(xt)
                xt_len = (xt != pad).sum(dim=1)  # (B,) - current sequence lengths
                position_indices = torch.arange(max_length, device=device).unsqueeze(0).expand(batch_size, -1)  # (B, L)
                confidence = 1.0 - (position_indices.float() / xt_len.unsqueeze(1).float().clamp(min=1))  # (B, L)
            
            elif confidence_method == "top_prob":
                # Top probability confidence
                token_logits = pred.token_logits  # (B, L, V)
                unmask_probs = F.softmax(token_logits, dim=-1)  # (B, L, V)
                confidence = unmask_probs.max(dim=-1)[0]  # (B, L)
            
            elif confidence_method == "prob_diff":
                # Probability difference confidence (top - second top)
                token_logits = pred.token_logits  # (B, L, V)
                unmask_probs = F.softmax(token_logits, dim=-1)  # (B, L, V)
                top2_probs, _ = torch.topk(unmask_probs, k=2, dim=-1)  # (B, L, 2)
                confidence = top2_probs[:, :, 0] - top2_probs[:, :, 1]  # (B, L)
            
            elif confidence_method == "entropy":
                # Entropy-based confidence (lower entropy = higher confidence)
                token_logits = pred.token_logits  # (B, L, V)
                unmask_probs = F.softmax(token_logits, dim=-1)  # (B, L, V)
                # Calculate entropy: -sum(p * log(p))
                entropy = -torch.sum(unmask_probs * torch.log(unmask_probs + 1e-10), dim=-1)  # (B, L)
                # Convert to confidence: lower entropy = higher confidence
                confidence = -entropy  # (B, L) - negative entropy so lower entropy gives higher confidence
            
            else:
                raise ValueError(f"Unknown confidence_method: {confidence_method}")
            
            # 3. Apply window constraint if enabled
            if use_sliding_window:
                # Vectorized window creation: calculate dynamic k for each batch
                k_values = torch.maximum(
                    torch.minimum(
                        (alpha * num_mask_positions).long(), 
                        torch.tensor(16, device=device
                    )
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
                
                # Get most likely tokens (need logits for token generation)
                if confidence_method == "position":
                    token_logits = pred.token_logits  # (B, L, V)
                most_likely_tokens = token_logits.argmax(dim=-1)  # (B, L)
                
                # Gather the tokens to place at selected positions
                selected_positions = all_top_indices[unmask_mask]  # Flattened valid positions
                batch_indices = torch.arange(batch_size, device=device).unsqueeze(1).expand(-1, max_unmask)[unmask_mask]  # Corresponding batch indices
                
                # Apply unmasking with sampled tokens
                new_xt[batch_indices, selected_positions] = most_likely_tokens[batch_indices, selected_positions]
        else:
            # Original tau-leaping unmask via Poisson
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
            
            # Don't insert within the prompt region
            if prompts is not None:
                # Create extended prompt mask for (L+1) positions
                extended_prompt_mask = torch.zeros((batch_size, max_length + 1), dtype=torch.bool, device=device)
                extended_pos_indices = torch.arange(max_length + 1, device=device).unsqueeze(0)
                extended_prompt_mask[:, :max_length] = extended_pos_indices[:, :max_length] < prompt_lens.unsqueeze(1)
                ext[extended_prompt_mask] = 0
            
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

            # Vectorized: Preserve the original prompt tokens
            if prompts is not None:
                xt_tmp[prompt_len_mask] = prompts[prompt_len_mask]

            # shift and scatter original tokens (excluding prompt region)
            new_pos_orig = pos_idx_L + ext_ex[:, :max_length]  # (B, L)
            orig_mask = pos_idx_L < xt_len.view(batch_size, 1)
            
            # Vectorized: Don't move prompt tokens
            if prompts is not None:
                orig_mask = orig_mask & ~prompt_len_mask
            
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


@torch.no_grad()
def generate_ours(
    model,
    prompt,
    tokenizer,
    steps: int,
    beta: float,
    model_type: str,
    temperature: float,
    remasking: str = 'random',
    mask_id: int = 126336,
    alpha: float = 5.0,  # Add alpha parameter
    confidence_method: str = "prob_diff",  # "position", "top_prob", "prob_diff", "entropy"
    use_sliding_window: bool = True,  # Add sliding window parameter
):
    pad_id = tokenizer.pad_token_id
    device = model.device
    B = prompt.size(0) if isinstance(prompt, torch.Tensor) else 1
    L = 1024
    dt = 1.0 / steps
    t = 0.0

    with torch.autocast(device_type="cuda"):
        return any_order_mask_insertion_tau_leaping_sampling(
            model, steps, mask_id, pad_id, B, L, 
            prompts=prompt, return_trace=True, alpha=alpha,
            confidence_method=confidence_method, use_sliding_window=use_sliding_window
        )


