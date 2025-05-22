import abc
from typing import Any
import torch
from torch import Tensor
from einops import rearrange
from dataclasses import dataclass

from schedule import Schedule


@dataclass
class ReparametrizedRate:
    per_token_posterior: Tensor
    length_posterior: Tensor


@dataclass
class Rate:
    unmask_rate: Tensor  # Shape [Batch, Length, Vocab]
    length_rate: Tensor  # Shape [Batch]


class Interpolant(abc.ABC):
    @abc.abstractmethod
    def sample_interpolant(self, t: Tensor, x1: Tensor) -> tuple[Any, Any]:
        """
        Return tuple (xt, st) of intermediate state and latent
        Shapes:
            x0: [B, N]
            x1: [B, N]
            t: [B]
        """
        raise NotImplementedError

    @abc.abstractmethod
    def reparametrised_conditional_rate(
        self, xt: Tensor, st: Tensor, t: Tensor, x1: Tensor
    ) -> ReparametrizedRate:
        raise NotImplementedError

    @abc.abstractmethod
    def to_actual_rate(self, rate: ReparametrizedRate, t: Tensor) -> Rate:
        return rate

    def conditional_rate(self, xt: Tensor, st: Tensor, t: Tensor, x1: Tensor) -> Rate:
        """
        Return aggregate conditional rate \sum_s' R((xt, st), (x', s')) for all x'
        Shape:
            x0, x1: [B, N]
        """
        reparametrized_rate = self.reparametrised_conditional_rate(xt, st, t, x1)
        return self.to_actual_rate(reparametrized_rate)


class SemiAutoregressiveInterpolant(Interpolant):
    def __init__(
        self,
        mask_schedule: Schedule,
        vocab_size: int,
        mask_token: int,
        pad_token: int,
        max_length: int,
    ):
        """
        TODO: Add knobs
        """
        super().__init__()
        assert 0 <= mask_token < vocab_size
        assert pad_token >= vocab_size
        self.mask_token = mask_token
        self.pad_token = pad_token
        self.max_length = max_length
        self.vocab_size = vocab_size
        self.mask_schedule = mask_schedule

    def sample_interpolant(self, t, x1) -> tuple[Tensor, Tensor]:
        B, L = x1.shape
        assert L == self.max_length

        id = torch.arange(L, device=x1.device).unsqueeze(0).expand(B, L)
        x1_lens = (x1 != self.pad_token).sum(dim=1)

        jumps = torch.rand(B, L, device=x1.device)
        jumps = torch.where(id < x1_lens.unsqueeze(1), jumps, 1.5)
        sorted, _ = torch.sort(jumps, dim=1)
        mask_proportion = self.mask_schedule.at(t)
        d = (sorted < mask_proportion.unsqueeze(1)).sum(dim=1)

        p = (t.unsqueeze(1) - sorted) / (1 - sorted)
        p = p.clamp(0.0, 1.0)
        samples = torch.bernoulli(p)
        samples = torch.where(samples == 1, x1, self.mask_token)
        xt = torch.where(id < d.unsqueeze(1), samples, self.pad_token)
        st = torch.where(id < d.unsqueeze(1), id, self.max_length + 1)

        return (xt, st)

    def reparametrised_conditional_rate(
        self, xt: Tensor, st: Tensor, t: Tensor, x1: Tensor
    ) -> ReparametrizedRate:
        B, L = x1.shape
        per_token_posterior = torch.zeros((B, L, self.vocab_size), device=x1.device)

        per_token_posterior.scatter_(
            2,
            # Filter pad_tokens
            rearrange(torch.where(xt != self.pad_token, x1, 0), "b l -> b l 1"),
            rearrange(
                torch.where(
                    xt == self.mask_token, torch.ones_like(t).unsqueeze(1), 0.0
                ),
                "b l -> b l 1",
            ),
        )
        x1_len = (x1 != self.pad_token).sum(dim=1)

        return ReparametrizedRate(
            per_token_posterior=per_token_posterior,
            length_posterior=torch.nn.functional.one_hot(
                x1_len, num_classes=self.max_length + 1
            ),
        )

    def to_actual_rate(self, xt: Tensor, rate: ReparametrizedRate, t: Tensor) -> Rate:
        rate_scale_factor = self.mask_schedule.rate_scale_factor(t)
        expected_len = (
            torch.arange(0, self.max_length + 1, device=xt.device).unsqueeze(0)
            * rate.length_posterior
        ).sum(dim=-1)
        xt_len = (xt != self.pad_token).sum(dim=1)
        return Rate(
            unmask_rate=rate.per_token_posterior / (1 - t).reshape(-1, 1, 1),
            length_rate=(expected_len - xt_len) * rate_scale_factor,
        )


class AnyOrderMaskInsertionInterpolant(Interpolant):
    def __init__(
        self,
        mask_schedule: Schedule,
        vocab_size: int,
        mask_token: int,
        pad_token: int,
        max_length: int,
    ):
        """
        TODO: Add knobs
        """
        super().__init__()
        assert 0 <= mask_token < vocab_size
        assert pad_token >= vocab_size
        self.mask_token = mask_token
        self.pad_token = pad_token
        self.max_length = max_length
        self.vocab_size = vocab_size
        self.mask_schedule = mask_schedule

    def sample_interpolant(self, t: Tensor, x1: Tensor) -> tuple[Tensor, Tensor]:
        B, L = x1.shape
        assert L == self.max_length

        # positional IDs
        id = torch.arange(L, device=x1.device).unsqueeze(0).expand(B, L)
        # true lengths (count of non-pad tokens)
        x1_lens = (x1 != self.pad_token).sum(dim=1)

        # 1) sample uniform noise, force padded positions to be “big”
        jumps = torch.rand(B, L, device=x1.device)
        jumps = torch.where(id < x1_lens.unsqueeze(1), jumps, 1.5)

        # 2) decide which positions survive: a random subset
        mask_proportion = self.mask_schedule.at(t)  # shape (B,)
        keep_mask = jumps < mask_proportion.unsqueeze(1)  # (B, L) bool
        d = keep_mask.sum(dim=1)  # how many to keep per example

        # 3) for each slot, flip a Bernoulli to choose actual token vs. mask
        p = (t.unsqueeze(1) - jumps) / (1 - jumps)
        p = p.clamp(0.0, 1.0)
        flips = torch.bernoulli(p)  # (B, L), 0/1
        samples = torch.where(flips == 1, x1, self.mask_token)

        # 4) pack all the 'kept' samples to the left, pad the rest
        #    Sorting the bool mask (True→1, False→0) brings all kept slots front
        keep_idx = keep_mask.argsort(dim=1, descending=True)  # (B, L)
        xt = torch.gather(samples, 1, keep_idx)  # (B, L)
        xt = torch.where(id < d.unsqueeze(1), xt, self.pad_token)

        # optionally, rebuild your positional‐index map the same way:
        st = torch.gather(id, 1, keep_idx)
        st = torch.where(id < d.unsqueeze(1), st, self.max_length + 1)

        return xt, st

    def reparametrised_conditional_rate(
        self, xt: Tensor, st: Tensor, t: Tensor, x1: Tensor, reparametrized=False
    ) -> ReparametrizedRate:
        B, L = x1.shape

        # --- 1) build the per-token unmask rates via st ---
        unmask_rate = torch.zeros((B, L, self.vocab_size), device=x1.device)

        # choose your base rate scalar
        rate = (
            torch.ones_like(t)
            if reparametrized
            else self.mask_schedule.rate_scale_factor(t)
        )

        # clamp any sentinel positions in st → L-1
        st_clamped = st.clamp(max=L - 1)  # so we can safely index x1

        # clamp x1 such that pad tokens does not result in out of bound errors
        x1_clamped = x1.clamp(max=self.vocab_size - 1)

        # gather the true token IDs that each xt-slot corresponds to
        orig_ids = torch.gather(x1_clamped, dim=1, index=st_clamped)  # (B, L)

        # compute, for each slot, whether it's a masked token
        mask_positions = xt == self.mask_token  # (B, L)

        # build the “src” values to scatter: rate for masked slots, else 0
        src = (rate.unsqueeze(1) * mask_positions.to(rate.dtype)).unsqueeze(
            2
        )  # (B, L, 1)

        # scatter into the vocab dimension at the original token IDs
        unmask_rate.scatter_(
            2,
            orig_ids.unsqueeze(2),  # (B, L, 1)
            src,
        )  # (B, L, 1)

        # --- 2) compute true vs current lengths & gaps as before ---
        x1_len = (x1 != self.pad_token).sum(dim=1)  # (B,)

        # sort+clamp st for gap-diffs
        sorted_st, _ = torch.sort(st, dim=1)
        sorted_clamped = torch.where(
            sorted_st <= x1_len.unsqueeze(1), sorted_st, x1_len.unsqueeze(1)
        )

        pad_front = x1.new_zeros((B, 1)) - 1
        pad_back = x1_len.unsqueeze(1)
        padded = torch.cat([pad_front, sorted_clamped, pad_back], dim=1)  # (B, L+2)
        gap_counts = padded[:, 1:] - padded[:, :-1] - 1  # (B, L+1)
        gap_counts = gap_counts.clamp(min=0)

        # one-hot each gap count → (B, L+1, max_length+1)
        length_posterior = torch.nn.functional.one_hot(
            gap_counts, num_classes=self.max_length + 1
        ).float()

        return ReparametrizedRate(
            per_token_posterior=unmask_rate, length_posterior=length_posterior
        )

    def to_actual_rate(self, xt: Tensor, rate: ReparametrizedRate, t: Tensor) -> Rate:
        B, L = xt.shape
        assert L == self.max_length

        device = xt.device

        # 1) per‑token unmask rates: just undo the (1−t) scaling
        #    rate.per_token_posterior has shape (B, L, V)
        unmask_rate = rate.per_token_posterior / (1 - t).view(B, 1, 1)

        # 2) compute expected gap‑counts from the gap‑wise one‑hot posterior
        #    rate.length_posterior has shape (B, L+1, max_length+1)
        idx = torch.arange(0, L + 1, device=device)  # (max_length+1,)
        # broadcast‑multiply and sum over the last axis → (B, L+1)
        expected_gaps = (rate.length_posterior * idx.view(1, 1, -1)).sum(dim=-1)

        # 3) scale by the schedule to get continuous‑time insertion rates
        rate_scale = self.mask_schedule.rate_scale_factor(t)  # (B,)
        length_rate = expected_gaps * rate_scale.view(B, 1)  # (B, L+1)

        return Rate(
            unmask_rate=unmask_rate,  # (B, L, V)
            length_rate=length_rate,  # (B, L+1)
        )
