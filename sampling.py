
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
    xt = pad * torch.ones_like((batch_size, max_length), device=device)

    dt = 1.0 / steps
    for i in range(steps):
        t = 1.0 - dt * i 
        unmask_rate, len_rate = model(xt, t)
        # Probabilistically unmask token

        # Fix diagonal entries by setting mask positions to negative sum of other rates
        mask_positions = (xt == mask)
        # First zero the mask token rate, so it's not included in the sum
        unmask_rate[mask_positions, mask] = 0
        # Then set it to negative sum of other rates
        unmask_rate[mask_positions, mask] = -unmask_rate[mask_positions, :].sum(dim=1)
        
        # Zero out rates for non-mask positions
        non_mask_positions = ~mask_positions
        unmask_rate[non_mask_positions] = 0

        # Approximate probability with Euler step 1 + Q dt
        trans_prob = unmask_rate * dt
        trans_prob.scatter_add_(2, xt.unsqueeze(-1), torch.ones_like(xt.unsqueeze(-1), dtype=unmask_rate.dtype))
        
        # Note that pad token is unchanged here
        new_xt = _sample_tokens(trans_prob.clamp(min=0, max=1))

        # Sample if next token should be added
        extension_prob = len_rate * dt
        extension = torch.bernoulli(extension_prob)
        
        # Find first pad position that will be replaced by mask
        first_pad_pos = torch.argmax((xt == pad).float(), dim=1)
    
        # Replace pad token with mask
        extend_mask = (first_pad_pos < max_length) & extension
        new_xt[extend_mask, first_pad_pos[extend_mask]] = mask

        xt = new_xt
        sampling_trace.append(new_xt)

    return xt, sampling_trace


