import torch
from dataclasses import dataclass
from torch import Tensor
import abc
from typing import Tuple
import abc
from omegaconf import DictConfig
import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional
import torch.nn.functional as F



@dataclass
class ModelPrediction:
    token_logits: Tensor
    length_posterior: Optional[Tensor]
    expected_gaps: Tensor

    def __init__(
        self,
        token_logits: Tensor,
        length_posterior: Optional[Tensor] = None,
        expected_gaps: Optional[Tensor] = None,
    ):
        assert length_posterior is not None or expected_gaps is not None
        self.token_logits = token_logits
        self.length_posterior = length_posterior
        self.expected_gaps = expected_gaps
        if self.expected_gaps is None:
            _, _, L = self.length_posterior.shape
            index = torch.arange(0, L, device=token_logits.device).view(1, 1, -1)
            self.expected_gaps = (self.length_posterior * index).sum(dim=-1)


@dataclass
class Rate:
    unmask_rate: Tensor  # Shape [Batch, Length, Vocab]
    length_rate: Tensor  # Shape [Batch]


@dataclass
class HittingTime:
    insertion_time: Tensor  # Shape [Batch, Length]
    unmasking_time: Tensor  # Shape [Batch, Length]

    def __iter__(self):
        yield from [self.insertion_time, self.unmasking_time]


@dataclass
class JointInterpolantResult:
    # Joint Interpolant
    xt: Tensor  # Shape [Batch, Length]
    st: Tensor  # Shape [Batch, Length]
    _x1: Tensor
    _pad_token: int
    _mask_token: int


    def to(self, device):
        self.xt = self.xt.to(device)
        self.st = self.st.to(device)
        self._x1 = self._x1.to(device)
        return self

    @property
    def mask_indices(self) -> Tensor:
        return self.xt == self._mask_token

    @property
    def unmasked(self) -> Tensor:
        return torch.gather(self._x1, 1, self.st)

    @property
    def xt_length(self) -> Tensor:
        # Calculate length of xt
        return (self.xt != self._pad_token).sum(dim=1)

    @property
    def x1_length(self) -> Tensor:
        # Calculate length of x1
        return (self._x1 != self._pad_token).sum(dim=1)

    @property
    def gaps_and_mask(self) -> tuple[Tensor, Tensor]:
        x1_len = self.x1_length
        gaps = self.st.clone()

        pad_front = gaps.new_zeros((gaps.shape[0], 1)) - 1  # -1 for the front padding
        pad_back = gaps.new_zeros((gaps.shape[0], 1))
        gaps = torch.cat([pad_front, gaps, pad_back], dim=1)  # Add a leading zero

        gaps.scatter_(
            1, self.xt_length.unsqueeze(1) + 1, x1_len.unsqueeze(1)
        )  # Fill the last position with x1_len

        gaps = gaps[:, 1:] - gaps[:, :-1] - 1
        gaps = torch.clamp(gaps, min=0)

        idx = torch.arange(gaps.size(1), device=self.xt.device).unsqueeze(
            0
        )  # shape [1, max_gap]
        mask = idx <= self.xt_length.unsqueeze(1)
        gaps[~mask] = 0

        return gaps, mask

def get_schedule_from_config(config: DictConfig):
    match config.type:
        case "geometric":
            return GeometricSchedule(min_val=config.min, max_val=config.max)
        case "linear":
            return LinearSchedule()
        case _:
            raise ValueError(f"Invalid schedule type: {config.type}")


class Schedule(abc.ABC):
    """
    Generic schedule class for masking or noising
    This represents function a : [0, 1] -> [0, 1] satisfying a(0) = 0, a(1) = 1 or at least approximately
    """

    @abc.abstractmethod
    def at(self, t: Tensor):
        """
        Return value a(t)
        """
        raise NotImplementedError

    @abc.abstractmethod
    def derivative_at(self, t: Tensor):
        """
        Return d/dt a(t)
        """
        raise NotImplementedError

    def rate_scale_factor(self, t: Tensor) -> Tensor:
        """
        Return d/dt a(t) / (1 - a(t)) common in rate matrix calculation
        """
        return self.derivative_at(t) / (1 - self.at(t))

    def sample(self, shape, device) -> Tensor:
        """
        Sample from the schedule, returns a tensor of shape `shape` with values in [0, 1]
        """
        uniform = torch.rand(shape, device=device)
        return self.inv(uniform)

    def sample_truncated(self, threshold, shape, device) -> Tensor:
        """
        Sample from a truncated schedule, returns a tensor of shape `shape` with values in [threshold, 1]
        """
        uniform = torch.rand(shape, device=device)
        return self.inv(uniform * (1 - threshold) + threshold)

    @abc.abstractmethod
    def inv(self, alpha: Tensor):
        """
        Given alpha in [0, 1] such that a(t)=alpha, returns the corresponding t.
        """
        raise NotImplementedError


class LinearSchedule(Schedule):
    def __init__(self):
        pass

    def at(self, t: Tensor):
        return t

    def derivative_at(self, t: Tensor):
        return torch.ones_like(t, device=t.device)

    def inv(self, alpha: Tensor):
        return alpha


class GeometricSchedule(Schedule, nn.Module):
    def __init__(self, min_val: float, max_val: float):
        super().__init__()
        self.register_buffer("min", Tensor([min_val]))
        self.register_buffer("max", Tensor([max_val]))

    def at(self, t: Tensor):
        min_val = self.min.to(t.device)
        max_val = self.max.to(t.device)
        return torch.exp(-(min_val ** (1 - t)) * max_val**t)

    def derivative_at(self, t):
        min_val = self.min.to(t.device)
        max_val = self.max.to(t.device)
        return (
            self.at(t)
            * min_val ** (1 - t)
            * max_val**t
            * (min_val.log() - max_val.log())
        )

    def inv(self, alpha: Tensor):
        log_min = self.min.to(alpha.device).log()
        log_max = self.max.to(alpha.device).log()
        return (torch.log(-torch.log(alpha)) - log_min) / (log_max - log_min)
    

class JointInterpolant(abc.ABC):
    def __init__(
        self,
        vocab_size: int,
        mask_token: int,
        pad_token: int,
        max_length: int,
    ):
        """
        TODO: Add knobs
        """
        self.mask_token = mask_token
        self.pad_token = pad_token
        self.max_length = max_length
        self.vocab_size = vocab_size

    @abc.abstractmethod
    def elbo_weight(self, t: Tensor, x1: Tensor):
        """
        Return the ELBO weight for the training, can be changed depends on the empirical results
        Shape:
            t: [B]
        Returns:
            weight_unmask: [B, L]
            weight_delete: [B, L+1]
        """
        raise NotImplementedError

    @abc.abstractmethod
    def to_actual_rate(self, prediction: ModelPrediction, t: Tensor) -> Rate:
        raise NotImplementedError

    @abc.abstractmethod
    def sample_interpolant(self, t: Tensor, x1: Tensor) -> JointInterpolantResult:
        """
        Sample the interpolant xt from x1 at time t
        Shapes:
            x1: [B, L]
            t: [B]
        Returns:
            xt: [B, L]
            st: [B, L] boolean mask of positions that corresponds to xt
            xt_mask_indices: [B, L] boolean mask of positions that are masked at xt
            x1_remained: [B, L] tokens that are not deleted, used for the training target
            gap_counts: [B, L+1] the number of deleted tokens between xt slots
        """
        raise NotImplementedError


class AnyOrderMaskInsertionInterpolant(JointInterpolant):
    def __init__(
        self,
        insertion_schedule: Schedule,
        unmask_schedule: Schedule,
        vocab_size: int,
        mask_token: int,
        pad_token: int,
        max_length: int,
    ):
        super().__init__(vocab_size, mask_token, pad_token, max_length)
        self.insertion_schedule = insertion_schedule
        self.unmask_schedule = unmask_schedule

    def hitting_time(self, t: Tensor, x1: Tensor) -> tuple[Tensor, Tensor]:
        """
        t1 is sampled from a uniform distribution over [0, 1]. when t1 < self.mask_schedule.at(t)
        t2 is sampled from a uniform distribution over [t1, 1]
        """
        B, L = x1.shape
        eps = 1e-6

        insert_time = self.insertion_schedule.sample((B, L), device=x1.device)
        insert_time = eps + (1 - eps) * insert_time  # ensure t1 is not 0
        unmask_time = self.unmask_schedule.sample_truncated(
            insert_time, (B, L), device=x1.device
        )

        return insert_time, unmask_time

    def elbo_weight(self, t: Tensor, x1: Tensor):
        """
        Return the ELBO weight for the training, can be changed depends on the empirical results
        """
        eps = 1e-6
        insert_weight = self.insertion_schedule.rate_scale_factor(t)
        insert_weight = insert_weight[:, None].expand(-1, x1.shape[1] + 1)

        unmask_weight = 1.0 / (1 - t + eps)
        unmask_weight = unmask_weight.unsqueeze(1).expand(-1, x1.shape[1])

        return unmask_weight.clone(), insert_weight.clone()

    def to_actual_rate(
        self, xt: Tensor, prediction: ModelPrediction, t: Tensor
    ) -> Rate:
        """
        Return the actual rate for the sampling
        Args:
            xt: [B, L] the sampled tokens
            prediction: ModelPrediction object containing token_posterior and expected_gaps
            t: [B] the time parameter
        """
        token_posterior = F.softmax(prediction.token_logits, dim=-1)  # (B, L, V)
        unmask_rate = token_posterior * self.unmask_schedule.rate_scale_factor(t).view(
            -1, 1, 1
        )
        length_rate = (
            prediction.expected_gaps
            * self.insertion_schedule.rate_scale_factor(t).view(-1, 1)
        )

        return Rate(
            unmask_rate=unmask_rate,  # (B, L, V)
            length_rate=length_rate,  # (B, L+1)
        )

    def sample_interpolant(self, t: Tensor, x1: Tensor, prompt_indices: Tensor) -> JointInterpolantResult:
        """
        Shapes:
            x1: [B, L]
            t: [B]
        Returns:
            xt: [B, L]
            st: [B, L] boolean mask of positions that corresponds to xt
            xt_mask_indices: [B, L] boolean mask of positions that are masked at xt
            x1_remained: [B, L] tokens that are not deleted, used for the training target
            gap_counts: [B, L+1] the number of deleted tokens between xt slots
        """
        # sample the stopping time (B, L, 2)
        insertion_time, unmasking_time = self.hitting_time(t, x1)

        clean_tokens = x1.ne(self.pad_token)
        deleted_tokens = clean_tokens & (t[:, None] < insertion_time) & (~prompt_indices)
        masked_tokens = (
            clean_tokens
            & (t[:, None] >= insertion_time)
            & (t[:, None] < unmasking_time)
        ) & (~prompt_indices)

        xt = torch.where(
            deleted_tokens,
            self.pad_token,  # for deletion, change to pad token
            torch.where(
                masked_tokens,
                self.mask_token,  # for masking, change to mask token
                x1,
            ),
        )

        st = xt.ne(self.pad_token).argsort(dim=1, descending=True, stable=True)
        xt = torch.gather(xt, 1, st)
        st[xt == self.pad_token] = 0

        return JointInterpolantResult(
            xt=xt, st=st, _x1=x1, _pad_token=self.pad_token, _mask_token=self.mask_token
        )


