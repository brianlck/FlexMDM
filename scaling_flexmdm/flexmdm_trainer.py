import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import Trainer
from transformers import DefaultDataCollator
import random
from tqdm import tqdm
import pickle
import torch.distributed as dist
from flexmdm_interpolant import AnyOrderMaskInsertionInterpolant, GeometricSchedule, LinearSchedule    

def jump_kernel_elbo(x, y, eps=1e-6):
    # x_safe: true length
    # y_safe: predicted length
    x_safe = torch.clamp(x, min=eps)
    y_safe = torch.clamp(y, min=eps)

    return y_safe - x_safe + x_safe * (torch.log(x_safe) - torch.log(y_safe))

def move_to_device(list_of_tensors, device):
    return [t.to(device) for t in list_of_tensors]


def vectorized_infill_process(input_ids, pad_token, prefix_cutoff, infill_tokens):
    """
    Performs a vectorized infill process on a batch of input IDs without loops.
    """
    batch_size, seq_len = input_ids.shape
    device = input_ids.device
    infill_tokens = torch.tensor(infill_tokens, dtype=input_ids.dtype, device=device)
    infill_len = len(infill_tokens)
    
    sample_lengths = (input_ids != pad_token).sum(dim=1)
    
    # Generate prefix_ends indices
    high_prefix_end = sample_lengths - 2
    prefix_range = high_prefix_end - prefix_cutoff
    rand_prefix_ends = torch.rand(batch_size, device=device) * prefix_range
    prefix_ends = (prefix_cutoff + rand_prefix_ends).long()
    
    # Generate suffix_starts indices
    low_suffix_start = prefix_ends + 1
    high_suffix_start = sample_lengths
    suffix_range = high_suffix_start - low_suffix_start
    rand_suffix_starts = torch.rand(batch_size, device=device) * suffix_range
    suffix_starts = (low_suffix_start + rand_suffix_starts).long()
    
    # Calculate new lengths and create the new tensor
    new_lengths = sample_lengths + 2 * infill_len
    new_sample = torch.full((batch_size, input_ids.shape[1]), pad_token, dtype=input_ids.dtype, device=device)
    
    # Create index tensors for both the new and original sequences
    new_indices = torch.arange(input_ids.shape[1], device=device).unsqueeze(0)
    orig_indices = torch.arange(seq_len, device=device).unsqueeze(0)

    # --- Copy prefix ---
    prefix_mask_orig = orig_indices < prefix_ends.unsqueeze(1)
    prefix_mask_new = new_indices < prefix_ends.unsqueeze(1)
    new_sample[prefix_mask_new] = input_ids[prefix_mask_orig]
    
    # --- Insert first infill token block ---
    infill_offsets = torch.arange(infill_len, device=device)
    insertion_indices = prefix_ends.unsqueeze(1) + infill_offsets
    new_sample.scatter_(1, insertion_indices, infill_tokens.repeat(batch_size, 1))

    # --- Copy middle part ---
    middle_indices_orig = (orig_indices >= prefix_ends.unsqueeze(1)) & (orig_indices < suffix_starts.unsqueeze(1))
    middle_indices_new = (new_indices >= prefix_ends.unsqueeze(1) + infill_len) & (new_indices < suffix_starts.unsqueeze(1) + infill_len)
    new_sample[middle_indices_new] = input_ids[middle_indices_orig]
    
    # --- Insert second infill token block ---
    insertion_indices = suffix_starts.unsqueeze(1) + infill_len + infill_offsets
    new_sample.scatter_(1, insertion_indices, infill_tokens.repeat(batch_size, 1))

    # --- Copy suffix ---
    suffix_indices_orig = (orig_indices >= suffix_starts.unsqueeze(1)) & (orig_indices < sample_lengths.unsqueeze(1))
    suffix_indices_new = (new_indices >= suffix_starts.unsqueeze(1) + 2 * infill_len) & (new_indices < new_lengths.unsqueeze(1))
    new_sample[suffix_indices_new] = input_ids[suffix_indices_orig]

    # Create vectorized prompt indices
    prompt_indices = (new_indices < prefix_ends.unsqueeze(1) + infill_len) | (new_indices >= suffix_starts.unsqueeze(1) + infill_len)
    prompt_indices = prompt_indices & (new_sample != pad_token)
    
    return new_sample, prompt_indices

class dLLMVariableLengthTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.label_names = ["input_ids"]

    def compute_loss(self, model, inputs, num_items_in_batch=None, return_outputs=False):
        """
        Variable Length Diffusion Loss Computation
        """
        result, elbo_weights, t = inputs.pop("interpolant_result"), inputs.pop("elbo_weights"), inputs.pop("t")
        if "length" in inputs:
            inputs.pop("length")
        if "prompt_indices" in inputs:
            inputs.pop("prompt_indices")
        if "prefix_cutoff" in inputs:
            inputs.pop("prefix_cutoff")
        
        masked_indices, x1_remained = result.mask_indices, result.unmasked
        gap_counts, len_loss_indices = result.gaps_and_mask
        unmask_weight, insert_weight = elbo_weights

        normalize_constant = 1024
        batch_size = x1_remained.shape[0]

        # device movement
        device = next(model.parameters()).device 
        masked_indices, x1_remained, gap_counts, len_loss_indices, unmask_weight, insert_weight, t = move_to_device(
            [masked_indices, x1_remained, gap_counts, len_loss_indices, unmask_weight, insert_weight, t], device
        )
        

        # model forward pass 
        out = model(timesteps = t, **inputs)
        logits, scalar_pred = out["logits"], out["length"]


        # compute the unmasking loss
        unmask_loss = unmask_weight[masked_indices] * F.cross_entropy(
            logits[masked_indices], x1_remained[masked_indices], reduction="none"
        )
        unmask_loss = unmask_loss.sum() / (batch_size * normalize_constant)

        # compute the length loss
        insertion_loss = insert_weight[len_loss_indices] * jump_kernel_elbo(
            gap_counts[len_loss_indices], scalar_pred[len_loss_indices])
        insertion_loss = insertion_loss.sum() / (batch_size * normalize_constant)
        
        scale = 1 / self.args.gradient_accumulation_steps
        loss = (unmask_loss + insertion_loss) * scale

        # log each loss at the end of the gradient accumulation step
        log_timing = (
            self.state.global_step % self.args.gradient_accumulation_steps == 0
        )
        unmask_mean = self.accelerator.gather(unmask_loss).mean()
        insertion_mean = self.accelerator.gather(insertion_loss).mean()

        if log_timing and self.accelerator.is_main_process:
            self.log(
                {
                    "unmask_loss": (unmask_mean).item(),
                    "insertion_loss": (insertion_mean).item()
                }
            )

        # for the evaluation loop, return_ouputs = True
        return loss if not return_outputs else (loss, logits)


class dLLMVariableDataCollator(DefaultDataCollator):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.max_length = kwargs["max_length"]
        self.mask_token_id = kwargs["tokenizer"].mask_token_id
        self.tokenizer = kwargs["tokenizer"]
        self.low_discrepancy = kwargs["low_discrepancy"]
        if kwargs["tokenizer"].mask_token_id is None:
            assert (
                "mask_token_id" in kwargs
            ), "For dLLM models, pass a mask_token_id or set it equal to tokenizer.mask_token_id"
            self.mask_token_id = kwargs["mask_token_id"]
        
        self.is_infill_task = kwargs.get("is_infill_task", False)
        self.infill_tokens = kwargs.get("infill_tokens", None)
        
        # Convert infill_token to list if it's not already
        if self.infill_tokens is not None and not isinstance(self.infill_tokens, list):
            self.infill_tokens = [self.infill_tokens]

        self.insertion_schedule = GeometricSchedule(min_val=10.0, max_val=0.01)
        self.unmasking_schedule = LinearSchedule()

        self.interpolant = AnyOrderMaskInsertionInterpolant(
            insertion_schedule=self.insertion_schedule,
            unmask_schedule=self.unmasking_schedule,
            vocab_size=kwargs["tokenizer"].vocab_size,
            mask_token = self.mask_token_id,
            pad_token = kwargs["tokenizer"].pad_token_id,
            max_length = self.max_length
        )

    def forward_process(self, batch, prompt_indices, eps=1e-3):
        input_ids = batch["input_ids"]
        B, _ = input_ids.shape
        if "t" not in batch:
            if self.low_discrepancy:
                if dist.is_initialized() and dist.is_available():
                    rank = dist.get_rank()
                    world_size = dist.get_world_size()
                    global_batch_size = B * world_size
                else:
                    rank = 0
                    global_batch_size = B

                intervals = torch.arange(B, device=input_ids.device, dtype=torch.float32) + rank * B
                offset = torch.rand(B, device=input_ids.device)
                t = (intervals + offset) / global_batch_size
            else:
                t = torch.rand((B,), device=input_ids.device)
        else:
            t = batch["t"]

        # sample the time step t: preventing blow-up
        t = (1 - eps) * t + eps # (B,)

        # sample from the interpolant
        interpolant_result = self.interpolant.sample_interpolant(t, input_ids, prompt_indices)


        # compute the ELBO weights
        elbo_weights = self.interpolant.elbo_weight(t, input_ids)

        return interpolant_result, elbo_weights, t

    def __call__(self, examples):
        # pad the examples to the max length
        for ex in examples:
            for key in ("input_ids", "attention_mask", "labels"):
                if key in ex and len(ex[key]) > self.max_length:
                    ex[key] = ex[key][: self.max_length]

        batch = self.tokenizer.pad(examples, 
            padding = "max_length",
            max_length = self.max_length,
            return_tensors = "pt"
        )

        if self.is_infill_task:
            input_ids, prompt_indices = vectorized_infill_process(
                batch["input_ids"],
                self.tokenizer.pad_token_id,
                batch["prefix_cutoff"],
                self.infill_tokens
            )
            batch["input_ids"] = input_ids
            batch["prompt_indices"] = prompt_indices
            
        # extract the prompt tokens
        elif "prompt_lengths" not in batch:
            # The case when there's no prompt
            prompt_indices = torch.zeros(batch["input_ids"].shape[0], self.max_length, dtype=torch.bool)
        else:
            prompt_lengths = batch.pop("prompt_lengths")
            suffix_lengths = batch.pop("suffix_length", None)
            prompt_lengths = prompt_lengths.unsqueeze(1) # (B, 1)
            positions = torch.arange(self.max_length) # (1, L)
            prompt_indices = (positions < prompt_lengths).bool() # (B, L)

        interpolant_result, elbo_weights, t = self.forward_process(batch, prompt_indices)
        batch["interpolant_result"] = interpolant_result
        batch["elbo_weights"] = elbo_weights
        batch["t"] = t
        batch["input_ids"] = interpolant_result.xt

        return batch

