import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import Trainer
from transformers import DefaultDataCollator
import random
from tqdm import tqdm
import pickle
import torch.distributed as dist
from sft_interpolant import AnyOrderMaskInsertionInterpolant, GeometricSchedule, LinearSchedule    

def jump_kernel_elbo(x, y, eps=1e-6):
    # x_safe: true length
    # y_safe: predicted length
    x_safe = torch.clamp(x, min=eps)
    y_safe = torch.clamp(y, min=eps)

    return y_safe - x_safe + x_safe * (torch.log(x_safe) - torch.log(y_safe))

def move_to_device(list_of_tensors, device):
    return [t.to(device) for t in list_of_tensors]


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

        # extract the prompt tokens
        if "prompt_lengths" not in batch:
            # The case when there's no prompt
            prompt_indices = torch.zeros(batch["input_ids"].shape[0], self.max_length, dtype=torch.bool)
        else:
            prompt_lengths = batch.pop("prompt_lengths")
            prompt_lengths = prompt_lengths.unsqueeze(1) # (B, 1)
            positions = torch.arange(self.max_length) # (1, L)
            prompt_indices = (positions < prompt_lengths).bool() # (B, L)

        interpolant_result, elbo_weights, t = self.forward_process(batch, prompt_indices)
        batch["interpolant_result"] = interpolant_result
        batch["elbo_weights"] = elbo_weights
        batch["t"] = t
        batch["input_ids"] = interpolant_result.xt

        return batch

        