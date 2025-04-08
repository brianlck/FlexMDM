
# TOOD: This is a very inflexible sampling algorithm -- Only works for semiautoregressive with one token addition at one time
# TODO: This code is quite bad, we'd like to refactor, can we use ein / einops?
import torch    


def semiauto_euler_sampling(model: torch.nn.Module, steps: int, mask: int, pad: int, batch_size: int, max_length: int):
    device = model.device
    xt = pad * torch.ones_like((batch_size, max_length), device=device)

    dt = 1.0 / steps
    for i in range(steps):
        t = 1.0 - dt * i 
        unmask_rate, len_rate = model(xt, t)

        # Probabilistically unmask token
        mask_positions = (xt == mask)
        other_rates_sum = torch.sum(unmask_rate[mask_positions, :mask], dim=1) + torch.sum(unmask_rate[mask_positions, mask+1:], dim=1)
        unmask_rate[mask_positions, mask] = -other_rates_sum
        
        # Zero out rates for non-mask positions
        non_mask_positions = ~mask_positions
        unmask_rate[non_mask_positions] = 0

        unmask_rate = unmask_rate * dt

        batch_indices = torch.arange(batch_size, device=device)
        pos_indices = torch.arange(max_length, device=device)
        xt_indices = xt[batch_indices.unsqueeze(1), pos_indices]
        unmask_rate[batch_indices.unsqueeze(1), pos_indices, xt_indices] += 1

        # Sample from categorical distribution for each position
        # Get probabilities for all mask positions at once
        mask_probs = torch.softmax(unmask_rate[mask_positions], dim=-1)
        # Sample from categorical distribution for all mask positions at once
        sampled_tokens = torch.multinomial(mask_probs, 1).squeeze(-1)
        # Update xt with sampled tokens at mask positions
        xt[mask_positions] = sampled_tokens
        
        extension_prob = len_rate * dt 
        extension = torch.bernoulli(extension_prob)
        # Find first pad token for each sequence in batch
        pad_mask = (xt == pad)
        first_pad_pos = torch.argmax(pad_mask.float(), dim=1)
        # Create mask for valid positions (where pad exists)
        valid_mask = (first_pad_pos < max_length)
        # Create batch indices and position indices for valid sequences
        batch_indices = torch.arange(batch_size, device=device)[valid_mask & extension]
        pos_indices = first_pad_pos[valid_mask & extension]
        # Set mask token at first pad position for valid sequences that extend
        xt[batch_indices, pos_indices] = mask


    


        


    
    