import torch    


def euler_sampling(model: torch.nn.Module, steps: int, mask: int, pad: int, batch_size: int, max_length: int):
    device = model.device
    xt = pad * torch.ones_like((batch_size, max_length), device=device)

    dt = 1.0 / steps
    for i in range(steps):
        t = 1.0 - dt * i 
        unmask_rate, len_rate = model(xt, t)
        
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

        


    
    