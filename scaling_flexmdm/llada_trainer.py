import torch
import torch.nn.functional as F
from transformers import Trainer
from transformers import DefaultDataCollator
import random
from tqdm import tqdm
import pickle
import torch.distributed as dist



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



class dLLMDataCollator(DefaultDataCollator):
    """
    Adds the forward noising process to the batch.
    Modify forward_process to change the noise schedule
    """

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.mask_token_id = kwargs["tokenizer"].mask_token_id
        self.tokenizer = kwargs["tokenizer"]
        if "max_length" in kwargs:
            self.max_length = kwargs["max_length"]
        if kwargs["tokenizer"].mask_token_id is None:
            assert (
                "mask_token_id" in kwargs
            ), "For dLLM models, pass a mask_token_id or set it equal to tokenizer.mask_token_id"
            self.mask_token_id = kwargs["mask_token_id"]
        
        # Optional infill parameters
        self.is_infill_task = kwargs.get("is_infill_task", False)
        if self.is_infill_task:
            self.instruction_tokens = kwargs.get("instruction_tokens", [])
            self.prefix_delimiters = kwargs.get("prefix_delimiters", [])
            self.suffix_delimiters = kwargs.get("suffix_delimiters", [])
            
            # Flatten delimiter lists if they are nested
            if self.prefix_delimiters and isinstance(self.prefix_delimiters[0], list):
                self.prefix_delimiters = [token for sublist in self.prefix_delimiters for token in sublist]
            if self.suffix_delimiters and isinstance(self.suffix_delimiters[0], list):
                self.suffix_delimiters = [token for sublist in self.suffix_delimiters for token in sublist]

    def forward_process(self, batch, eps=1e-3):
        input_ids = batch["input_ids"]
        B, N = input_ids.shape
        if "t" not in batch:
            t = torch.rand((B,), device=input_ids.device)
        else:
            t = batch["t"]

        t = (1 - eps) * t + eps
        t = t[:, None].repeat(1, N)

        mask_indices = torch.rand((B, N), device=input_ids.device) < t
        noisy_batch = torch.where(mask_indices, self.mask_token_id, input_ids)
        return noisy_batch, t, mask_indices

    def __call__(self, batch):
        batch = super().__call__(batch)

        # Pad input_ids to max_length before processing
        if hasattr(self, 'max_length'):
            batch = self.tokenizer.pad(batch, 
                padding = "max_length",
                max_length = self.max_length,
                return_tensors = "pt"
            )

            print(batch["input_ids"].shape)

        batch["labels"] = batch["input_ids"].clone()
        # Apply infill transformation if enabled
        if self.is_infill_task:
            batch["input_ids"], infill_prompt_indices = vectorized_infill_process(
                batch["input_ids"],
                self.tokenizer.pad_token_id,
                batch["prefix_cutoff"],
                self.instruction_tokens,
                self.prefix_delimiters,
                self.suffix_delimiters
            )
            batch["labels"] = batch["input_ids"].clone()
        
        noisy_batch, batch["t"], mask_indices = self.forward_process(batch)
        batch["labels"][~mask_indices] = -100
        batch["num_prompt_tokens"] = 0
        
        if "prompt_lengths" in batch:
            prompt_lengths = batch.pop("prompt_lengths")
            prompt_lengths = prompt_lengths.unsqueeze(1) # (B, 1)
            prompt_length_indices = torch.arange(noisy_batch.shape[1]).unsqueeze(0) # (1, L)

            # mask the prompt tokens
            prompt_mask = prompt_length_indices < prompt_lengths # (B, L)
            noisy_batch[prompt_mask] = batch["input_ids"][prompt_mask].clone()
            batch["labels"][prompt_mask] = -100
            batch["num_prompt_tokens"] = prompt_mask.sum()
        elif self.is_infill_task:
            # Use infill prompt indices to mask prompt tokens
            noisy_batch[infill_prompt_indices] = batch["input_ids"][infill_prompt_indices].clone()
            batch["labels"][infill_prompt_indices] = -100
            batch["num_prompt_tokens"] = infill_prompt_indices.sum()
            
        batch["input_ids"] = noisy_batch.long()
        return batch

class dLLMTrainer(Trainer):
    def compute_loss(self, model, inputs, num_items_in_batch=None, return_outputs=False):
        """
        Absorbing state diffusion loss computation
        NOTE: time step t here is different from ours
        """
        normalize_constant = 4096
        batch_size = inputs["input_ids"].size(0)

        labels, t, num_prompt_tokens = inputs.pop("labels"), inputs.pop("t"), inputs.pop("num_prompt_tokens")

        if "prefix_cutoff" in inputs:
            inputs.pop("prefix_cutoff")
        
        outputs = model(**inputs)
        logits = outputs.logits
        unscaled_loss = F.cross_entropy(
            logits.view(-1, logits.shape[-1]), labels.view(-1), reduction="none"
        ).view(logits.shape[0], -1)
        if (self.state.global_step + 1) % self.args.logging_steps == 0:
            self.log({"unscaled_loss": (unscaled_loss.sum() / (labels != -100).sum()).item()})
        loss = unscaled_loss / t
        loss = loss.sum() / (batch_size * normalize_constant)
        
        # double-check debug
        if return_outputs:
            print("Retuning outputs")
            return loss, {"dummy": None}

        return loss


class dLLMSFTDataset(torch.utils.data.Dataset):
    """
    Similar to AR datasets, except in inference, we keep the timsteps fixed
    """

    def __init__(self, data, tokenizer, max_length, eval=False):
        super().__init__()
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.eval = eval
        if self.eval:
            self.t = torch.linspace(0, 1, len(self.data))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        out = self.data[idx]
        if self.eval:
            out["t"] = self.t[idx]
        return out


class dLLMDataCollator(DefaultDataCollator):
    """
    Adds the forward noising process to the batch.
    Modify forward_process to change the noise schedule
    """

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.mask_token_id = kwargs["tokenizer"].mask_token_id
        self.tokenizer = kwargs["tokenizer"]
        if "max_length" in kwargs:
            self.max_length = kwargs["max_length"]
        if kwargs["tokenizer"].mask_token_id is None:
            assert (
                "mask_token_id" in kwargs
            ), "For dLLM models, pass a mask_token_id or set it equal to tokenizer.mask_token_id"
            self.mask_token_id = kwargs["mask_token_id"]
        
        # Optional infill parameters
        self.is_infill_task = kwargs.get("is_infill_task", False)
        if self.is_infill_task:
            self.instruction_tokens = kwargs.get("instruction_tokens", [])
            self.prefix_delimiters = kwargs.get("prefix_delimiters", [])
            self.suffix_delimiters = kwargs.get("suffix_delimiters", [])

    def forward_process(self, batch, eps=1e-3):
        input_ids = batch["input_ids"]
        B, N = input_ids.shape
        if "t" not in batch:
            t = torch.rand((B,), device=input_ids.device)
        else:
            t = batch["t"]

        t = (1 - eps) * t + eps
        t = t[:, None].repeat(1, N)

        mask_indices = torch.rand((B, N), device=input_ids.device) < t
        noisy_batch = torch.where(mask_indices, self.mask_token_id, input_ids)
        return noisy_batch, t, mask_indices

    def __call__(self, batch):
        batch = super().__call__(batch)

        # Pad input_ids to max_length before processing
        if hasattr(self, 'max_length'):
            batch = self.tokenizer.pad(batch, 
                padding = "max_length",
                max_length = self.max_length,
                return_tensors = "pt"
            )

        batch["labels"] = batch["input_ids"].clone()
        # Apply infill transformation if enabled
        if self.is_infill_task:
            batch["input_ids"], infill_prompt_indices = vectorized_infill_process(
                batch["input_ids"],
                self.tokenizer.pad_token_id,
                batch["prefix_cutoff"],
                self.instruction_tokens,
                self.prefix_delimiters,
                self.suffix_delimiters
            )
            batch["labels"] = batch["input_ids"].clone()
        
        noisy_batch, batch["t"], mask_indices = self.forward_process(batch)
        batch["labels"][~mask_indices] = -100
        batch["num_prompt_tokens"] = 0
        
        if "prompt_lengths" in batch:
            prompt_lengths = batch.pop("prompt_lengths")
            prompt_lengths = prompt_lengths.unsqueeze(1) # (B, 1)
            prompt_length_indices = torch.arange(noisy_batch.shape[1]).unsqueeze(0) # (1, L)

            # mask the prompt tokens
            prompt_mask = prompt_length_indices < prompt_lengths # (B, L)
            noisy_batch[prompt_mask] = batch["input_ids"][prompt_mask].clone()
            batch["labels"][prompt_mask] = -100
            batch["num_prompt_tokens"] = prompt_mask.sum()
        elif self.is_infill_task:
            # Use infill prompt indices to mask prompt tokens
            noisy_batch[infill_prompt_indices] = batch["input_ids"][infill_prompt_indices].clone()
            batch["labels"][infill_prompt_indices] = -100
            batch["num_prompt_tokens"] = infill_prompt_indices.sum()
            
        batch["input_ids"] = noisy_batch.long()
        return batch


SYSTEM_PROMPT = """
Respond in the following format:
<reasoning>
Your reasoning here
</reasoning>
<answer>
...
</answer>
"""


def preprocess_dataset(data, tokenizer, max_length, test_split=0.01):
    preprocessed_data = []
    # TODO: check if the pad_token = mask_token
    for i in tqdm(range(len(data)), desc="Preprocessing dataset"):
        question = SYSTEM_PROMPT + "\n\n" + data[i]["question"]
        trajectory = f"<reasoning>{data[i]['thinking_trajectories'][0]}</reasoning>\n<answer>{data[i]['attempt']}</answer>"
        prompt = [{"role": "user", "content": question}]
        response = [{"role": "assistant", "content": trajectory}]
        inputs = tokenizer.apply_chat_template(prompt + response, tokenize=False)
        prompt = tokenizer.apply_chat_template(prompt, tokenize=False) + "\n"
        tokenized_input = tokenizer(
            inputs, return_tensors="pt", truncation=True, max_length=max_length, padding="max_length"
        ).input_ids.squeeze(0)
        num_tokens = tokenized_input.shape[0]
        tokenized_prompt = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length)
        preprocessed_data.append(
            {
                "input_ids": tokenized_input,
                "prompt_lengths": tokenized_prompt.attention_mask.sum(-1),
            }
        )

    random.shuffle(preprocessed_data)
    test_data = preprocessed_data[: int(len(preprocessed_data) * test_split)]
    train_data = preprocessed_data[int(len(preprocessed_data) * test_split) :]
    return train_data, test_data
