
# TOOD: This is a very inflexible sampling algorithm -- Only works for semiautoregressive with one token addition at one time
# TODO: This code is quite bad, we'd like to refactor, can we use ein / einops?
import torch    

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

def semiauto_euler_sampling(model: torch.nn.Module, steps: int, mask: int, pad: int, batch_size: int, max_length: int):
    device = model.device
    xt = pad * torch.ones((batch_size, max_length), device=device).to(torch.int64)
    sampling_trace = []

    dt = 1.0 / steps
    t = torch.ones((batch_size,), device=device)
    for i in range(steps):
        t = t - dt
        unmask_rate, len_rate = model(xt, t)
        unmask_rate = unmask_rate / (1 - t).reshape(-1, 1, 1)
        len_rate = len_rate / (1 - t) * max_length
        # Probabilistically unmask token

        # Fix diagonal entries by setting mask positions to negative sum of other rates
        mask_positions = (xt == mask).nonzero(as_tuple=True)
        
        # Zero out rates for non-mask positions
        unmask_rate[xt != mask] = 0

        # First zero the mask token rate, so it's not included in the sum
        unmask_rate[*mask_positions,  mask] = 0
        # Then set it to negative sum of other rates
        unmask_rate[*mask_positions, mask] = -unmask_rate[*mask_positions, :].sum(dim=1)

        # Approximate probability with Euler step 1 + Q dt
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
        xt = new_xt
        sampling_trace.append(new_xt)

    return xt, sampling_trace


