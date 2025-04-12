import abc
from typing import Any
import torch
from torch import Tensor
from einops import rearrange
from dataclasses import dataclass

from schedule import Schedule


@dataclass
class SemiAutoregressiveRate:
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
    def conditional_rate(self, xt: Tensor, st: Tensor, t: Tensor, x1: Tensor, transformed: bool = False) -> Any:
        """
        Return aggregate conditional rate \sum_s' R((xt, st), (x', s')) for all x'
        Shape:
            x0, x1: [B, N]
        """
        raise NotImplementedError

    @abc.abstractmethod
    def to_actual_rate(self, rate, t):
        return rate


class SemiAutoregressiveInterpolant(abc.ABC):
    def __init__(self, mask_schedule: Schedule, vocab_size: int, mask_token: int, pad_token: int, max_length: int):
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

    def conditional_rate(self, xt, st, t, x1, transformed=False) -> SemiAutoregressiveRate:
        B, L = x1.shape
        unmask_rate = torch.zeros((B, L, self.vocab_size), device=x1.device)
        if transformed:
            rate = torch.ones_like(t)
        else:
            rate = self.mask_schedule.rate_scale_factor(t)
        
        unmask_rate.scatter_(
            2,
            # Filter pad_tokens
            rearrange(torch.where(xt != self.pad_token, x1, 0), "b l -> b l 1"),
            rearrange(torch.where(xt == self.mask_token, rate.unsqueeze(1), 0.0), "b l -> b l 1"),
        )
        x1_len = (x1 != self.pad_token).sum(dim=1)
        xt_len = (xt != self.pad_token).sum(dim=1)

        if transformed:
            len_rates = torch.nn.functional.one_hot(x1_len, num_classes=self.max_length + 1)
        else:
            len_rates = (x1_len - xt_len) * self.mask_schedule.rate_scale_factor(t)

        return SemiAutoregressiveRate(unmask_rate=unmask_rate, length_rate=len_rates)

    def to_actual_rate(self, xt: Tensor, rate: SemiAutoregressiveRate, t: Tensor) -> SemiAutoregressiveRate:
        rate_scale_factor = self.mask_schedule.rate_scale_factor(t)
        expected_len = (torch.arange(0, self.max_length+1, device=xt.device).unsqueeze(0) * rate.length_rate).sum()
        xt_len = (xt != self.pad_token).sum(dim=1)
        return SemiAutoregressiveRate(
            unmask_rate=rate.unmask_rate * rate_scale_factor,
            length_rate=(expected_len - xt_len) * rate_scale_factor
        )
