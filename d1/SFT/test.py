import torch



def one_infill_process(input_ids, pad_token, prefix_end, suffix_start, instruction_tokens, prefix_delimiters, suffix_delimiters):
    """
    Performs a single fill-in-the-middle (FIM) transformation on a sequence.

    This function takes a sequence of token IDs and reformats it for a text infilling task.
    It rearranges the sequence into the FIM format:
    `[instruction_tokens] [prefix_delimiters[0]] [prefix] [prefix_delimiters[1]] [suffix_delimiters[0]] [suffix] [suffix_delimiters[1]] [middle]`
    where:
    - `instruction_tokens`: Special instruction tokens at the beginning
    - `prefix`: The part of the sequence before the selected span
    - `middle`: The selected span of text to be "filled in"
    - `suffix`: The part of the sequence after the selected span
    - `prefix_delimiters`: Pair of tokens that wrap the prefix
    - `suffix_delimiters`: Pair of tokens that wrap the suffix

    Args:
        input_ids (torch.Tensor): A sequence of input token IDs.
        pad_token (int): The ID of the padding token.
        prefix_end (int): The end index for the prefix.
        suffix_start (int): The start index for the suffix.
        instruction_tokens (list[int]): A list of instruction token IDs at the beginning.
        prefix_delimiters (list[int]): A list of two token IDs to wrap the prefix.
        suffix_delimiters (list[int]): A list of two token IDs to wrap the suffix.

    Returns:
        tuple[torch.Tensor, torch.Tensor]:
        - new_sample (torch.Tensor): The transformed sequence with the FIM format.
        - prompt_indices (torch.Tensor): A boolean mask indicating the prompt tokens in `new_sample`.
    """
    device = input_ids.device

    instruction_len = len(instruction_tokens)
    prefix_open_len = len(prefix_delimiters[0])
    prefix_close_len = len(prefix_delimiters[1])
    suffix_open_len = len(suffix_delimiters[0])
    suffix_close_len = len(suffix_delimiters[1])

    input_len = (input_ids != pad_token).sum()

    instruction_tokens = torch.tensor(instruction_tokens, dtype=input_ids.dtype, device=device)
    prefix_open_delim = torch.tensor(prefix_delimiters[0], dtype=input_ids.dtype, device=device)
    prefix_close_delim = torch.tensor(prefix_delimiters[1], dtype=input_ids.dtype, device=device)
    suffix_open_delim = torch.tensor(suffix_delimiters[0], dtype=input_ids.dtype, device=device)
    suffix_close_delim = torch.tensor(suffix_delimiters[1], dtype=input_ids.dtype, device=device)

    new_sample = torch.full((input_ids.shape[0],), pad_token, dtype=input_ids.dtype, device=device)
    new_sample[:instruction_len] = instruction_tokens
    new_sample[instruction_len:instruction_len + prefix_open_len] = prefix_open_delim
    new_sample[instruction_len + prefix_open_len:instruction_len + prefix_open_len + prefix_end] = input_ids[:prefix_end]
    new_sample[instruction_len + prefix_open_len + prefix_end:instruction_len + prefix_open_len + prefix_end + prefix_close_len] = prefix_close_delim

    suffix_offset = instruction_len + prefix_open_len + prefix_end + prefix_close_len
    new_sample[suffix_offset:suffix_offset + suffix_open_len] = suffix_open_delim
    new_sample[suffix_offset + suffix_open_len:suffix_offset + suffix_open_len + (input_len - suffix_start)] = input_ids[suffix_start:input_len]
    new_sample[suffix_offset + suffix_open_len + (input_len - suffix_start):suffix_offset + suffix_open_len + (input_len - suffix_start) + suffix_close_len] = suffix_close_delim

    middle_start = suffix_offset + suffix_open_len + (input_len - suffix_start) + suffix_close_len

    new_sample[middle_start:middle_start + (suffix_start - prefix_end)] = input_ids[prefix_end:suffix_start]
    prompt_indices = torch.ones_like(new_sample, dtype=torch.bool)
    prompt_indices[:middle_start] = True
    prompt_indices[middle_start:] = False

    return new_sample, prompt_indices




print(one_infill_process(    torch.tensor([1, 1, 1, 1, 2, 9, 4, 5, 6, 7, 8, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]),
    pad_token=3,
    prefix_end=5,
    suffix_start=8,
    instruction_tokens=[0, 2],
    prefix_delimiters=[[99, 999], [9999, 99999]],
    suffix_delimiters=[[6, 1], [7, 11]]
))



def vectorized_infill_process(input_ids, pad_token, prefix_cutoff, instruction_tokens, prefix_delimiters, suffix_delimiters):
    batch_size, _ = input_ids.shape
    device = input_ids.device

    input_lengths = (input_ids != pad_token).sum(dim=1)
    prefix_range = input_lengths - prefix_cutoff
    rand_prefix_ends = torch.rand(batch_size, device=device) * prefix_range
    prefix_ends = (prefix_cutoff + rand_prefix_ends).long()
    
    # Generate suffix_starts indices
    low_suffix_start = prefix_ends + 1
    high_suffix_start = input_lengths
    suffix_range = high_suffix_start - low_suffix_start
    rand_suffix_starts = torch.rand(batch_size, device=device) * suffix_range
    suffix_starts = (low_suffix_start + rand_suffix_starts).long()
    
    new_samples, prompt_indices = [], []
    for i in range(batch_size):
        new_sample, prompt_index = one_infill_process(
            input_ids[i],
            pad_token,
            prefix_ends[i],
            suffix_starts[i],
            instruction_tokens,
            prefix_delimiters,
            suffix_delimiters
        )
        new_samples.append(new_sample)
        prompt_indices.append(prompt_index)

    new_samples = torch.stack(new_samples)
    prompt_indices = torch.stack(prompt_indices)

    return new_samples, prompt_indices


print(vectorized_infill_process(
    torch.tensor([[1, 1, 1, 1, 2, 9, 4, 5, 6, 7, 8, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]]),
    pad_token=3,
    prefix_cutoff=2,
    instruction_tokens=[0, 2],
    prefix_delimiters=[[99, 999], [9999, 99999]],
    suffix_delimiters=[[6, 1], [7, 11]]
))