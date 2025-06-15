import abc
import torch
import torch.nn as nn
from torch import Tensor


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

    def rate_scale_factor(self, t: Tensor):
        """
        Return d/dt a(t) / (1 - a(t)) common in rate matrix calculation
        """
        return self.derivative_at(t) / (1 - self.at(t))


class LinearSchedule(Schedule):
    def __init__(self):
        pass

    def at(self, t: Tensor):
        return t

    def derivative_at(self, t: Tensor):
        return torch.ones_like(t, device=t.device)



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


class NoSchedule(Schedule):
    """
    Ghost schedule for the vanilla MDM
    """
    def __init__(self):
        pass
    
    def at(self, t: Tensor):
        return torch.ones_like(t, device=t.device)

    def derivative_at(self, t: Tensor):
        return None