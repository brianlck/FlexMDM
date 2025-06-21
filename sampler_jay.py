from train import TransdimensionalFlowModule
from train_MDM import MDM
from data.text import decode_sequence_with_mask
import torch
from typing import Any, List
from dataclasses import dataclass
import matplotlib.pyplot as plt
from typing import Tuple


@dataclass
class SamplingResult:
    samples: torch.Tensor
    # Trace is supposed to be processed sequentially as updates are not commutative
    # trace: Optional[list[SamplingTraceDatapoint]]

    def __iter__(self):
        yield from [self.samples, self.trace]


# ------------------------------------------------------------
# Util functions for sampling
# ------------------------------------------------------------


def extract_non_pad(xs: torch.Tensor, pad: int) -> List[List[int]]:
    result = []
    for seq in xs:
        non_pad = seq[seq != pad].tolist()
        result.append(non_pad)
    return result


def model_load(model_name: str):
    if model_name == "vlmdm":
        checkpoint_path = "checkpoints/wikitext2/any_order/20250613-192826/epoch-epoch=505-val_loss-val_loss=1.8743.ckpt"
        model = TransdimensionalFlowModule.load_from_checkpoint(checkpoint_path)
    elif model_name == "mdm":
        checkpoint_path = "checkpoints/wikitext2/vanilla_MDM/20250613-192949/epoch-epoch=73-val_loss-val_loss=1.7349.ckpt"
        model = MDM.load_from_checkpoint(checkpoint_path)
    else:
        raise ValueError(f"Model {model_name} not found")
    print("Model loaded")
    return model


def unmasking_rate_calc(s: float, t: float) -> float:
    return (s - t) / (1 - t) if s < 1 - 1e-6 else 1.0


def insertion_rate_calc(s: float, t: float) -> float:
    # for the last step, we set the insertion rate to 0
    # this ensures that our sampling process ends up with clean tokens
    return (
        (1 - s) / (1 - t) * (torch.log(1 - t) - torch.log(1 - s))
        if s < 1 - 1e-6
        else 0.0
    )


def add_gumbel_noise(logits: torch.Tensor, mask, temperature=1.0) -> torch.Tensor:
    """
    As suggested by https://arxiv.org/pdf/2409.02908, we use float64 for the gumbel max method.
    """
    logits = logits.to(torch.float64)
    logits[:, mask] = float("-inf")
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (-torch.log(noise)) ** temperature

    return logits.exp() / gumbel_noise


def valid_insertion_positions(xt: torch.Tensor, pad: int) -> torch.Tensor:
    not_pad_tokens = xt != pad  # B X L
    left_pad_tensor = torch.ones((xt.shape[0], 1), dtype=torch.bool, device=xt.device)
    valid_positions = torch.cat([left_pad_tensor, not_pad_tokens], dim=1)  # B X (L+1)
    return valid_positions


def length(xt: torch.Tensor, pad: int, mask: int) -> torch.Tensor:
    seq_len = (xt != pad).sum(dim=1)
    clean_tokens = ((xt != mask) & (xt != pad)).sum(dim=1)
    return seq_len, clean_tokens


# ------------------------------------------------------------
# Sampler for our model
# ------------------------------------------------------------

# TODO: (for a future note) can we control the discretization space, make the intial time steps less fine (we anyway insert mask tokens) / make the final time steps more fine?


@torch.no_grad()
def mdm_style_sampling(
    model: torch.nn.Module,
    num_steps: int,
    tokeniser: Any,
    mask: int,
    pad: int,
    batch_size: int,
    max_length: int,
    temperature: float = 1.0,
):
    device = model.device
    B = batch_size
    L = max_length

    # precompute
    pos_idx = torch.arange(L, device=device).view(1, L).expand(B, L)

    # intialize all-pad sequencne and trace
    xt = torch.full((B, L), pad, dtype=torch.int64, device=device)
    # xt[:, 0:3] = mask # all-pad sentence does not lead to meaningful output
    sampling_trace = [xt]
    len_trace = []

    for i in range(num_steps):
        seq_len, clean_tokens = length(xt, pad, mask)
        len_trace.append((i, seq_len, clean_tokens))

        t = torch.tensor(i / num_steps, device=device)
        s = torch.tensor((i + 1) / num_steps, device=device)
        xs_tmp = xt.clone()

        t_input = t.unsqueeze(0)
        pred_rate = model(xt, t_input)

        # --- select unmask indices ---
        unmasking_rate = unmasking_rate_calc(s, t)
        mask_indices = xt == mask
        rand = torch.rand_like(xt, dtype=torch.float32, device=device)
        unmasking_indices = mask_indices & (
            rand < unmasking_rate
        )  # select with index to unmask
        b_ids, p_ids = torch.where(unmasking_indices)
        logits_with_noise = add_gumbel_noise(
            pred_rate.token_posterior[b_ids, p_ids], mask, temperature
        )  # add gumbel noise to the logits
        sampled_tokens = torch.argmax(logits_with_noise, dim=-1)
        xs_tmp[b_ids, p_ids] = sampled_tokens

        # --- select insertion indices ---
        # 1) sample gaps to extend
        insertion_rate = insertion_rate_calc(s, t)
        valid_insertion_pos = valid_insertion_positions(xs_tmp, pad)  # (B, L+1)
        insertion_probs = (pred_rate.expected_gaps * insertion_rate).clamp(0.0, 1.0)
        bern_samples = (
            torch.bernoulli(insertion_probs).to(torch.int64) * valid_insertion_pos
        )  # (B, L+1)

        # 2) calculate the positions of new masked tokens and original tokens
        ext_ex = bern_samples.cumsum(dim=1)  # (B , L+1)
        new_pos_orig = pos_idx + ext_ex[:, :L]

        # 3) calculate the final length
        origin_lens = (xs_tmp != pad).sum(dim=1)
        insert_lens = bern_samples.sum(dim=1)
        final_lens = (origin_lens + insert_lens).clamp(max=L)

        # 4) scatter
        xs = torch.full((B, L), mask, dtype=torch.int64, device=device)
        valid = new_pos_orig < L
        b_idx, p_idx = torch.nonzero(valid, as_tuple=True)
        v_idx = new_pos_orig[b_idx, p_idx]
        xs[b_idx, v_idx] = xs_tmp[b_idx, p_idx]

        # 5) padding
        poss_all = pos_idx
        pad_positions = poss_all >= final_lens.view(B, 1)
        xs[pad_positions] = pad
        xt = xs

        sampling_trace.append(
            decode_sequence_with_mask(extract_non_pad(xt, pad), tokeniser, pad, mask)
        )

    return sampling_trace, len_trace


# ------------------------------------------------------------
# Sampling for MDM
# ------------------------------------------------------------


@torch.no_grad()
def mdm_sampling(
    model: torch.nn.Module,
    num_steps: int,
    tokeniser: Any,
    mask: int,
    pad: int,
    batch_size: int,
    max_length: int,
    temperature: float = 1.0,
):
    device = model.device
    B = batch_size
    L = max_length

    # for MDM, we start with all-mask sequence
    xt = torch.full((B, L), mask, dtype=torch.int64, device=device)
    sampling_trace = [xt]
    len_trace = []

    for i in range(num_steps):
        seq_len, clean_tokens = length(xt, pad, mask)
        len_trace.append((i, seq_len, clean_tokens))

        t = torch.tensor(i / num_steps, device=device)
        s = torch.tensor((i + 1) / num_steps, device=device)
        xs_tmp = xt.clone()

        t_input = t.unsqueeze(0)
        pred_rate = model(xt, t_input)

        # --- select unmask indices ---
        unmasking_rate = unmasking_rate_calc(s, t)
        mask_indices = xt == mask
        rand = torch.rand_like(xt, dtype=torch.float32, device=device)
        unmasking_indices = mask_indices & (
            rand < unmasking_rate
        )  # select with index to unmask
        b_ids, p_ids = torch.where(unmasking_indices)
        logits_with_noise = add_gumbel_noise(
            pred_rate.token_posterior[b_ids, p_ids], mask, temperature
        )  # add gumbel noise to the logits
        sampled_tokens = torch.argmax(logits_with_noise, dim=-1)
        xs_tmp[b_ids, p_ids] = sampled_tokens
        xt = xs_tmp

        sampling_trace.append(
            decode_sequence_with_mask(extract_non_pad(xt, pad), tokeniser, pad, mask)
        )

    return sampling_trace, len_trace


# ------------------------------------------------------------
# Plotting functions
# ------------------------------------------------------------


def plot_len_trace(len_trace: List[Tuple[int, int, int]], model_name: str):
    plt.figure(figsize=(10, 5))
    steps, seq_lens, clean_tokens = zip(*len_trace)
    plt.plot(steps, seq_lens, label="Sequence length")
    plt.plot(steps, clean_tokens, label="Number of clean tokens")
    plt.xlabel("Generation step")
    plt.ylabel("Number")
    plt.title("Quantity changes over generation steps")
    plt.legend()
    plt.savefig(f"len_trace_{model_name}.png")
    plt.close()
