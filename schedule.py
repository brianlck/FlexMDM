
import abc
import torch
import torch.nn as nn
from torch import Tensor    

class Schedule(abc.abstractmethod):
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
    def derivative_at(self, t:  Tensor):
        """
            Return d/dt a(t)  
        """
        raise NotImplementedError

    @abc.abstractmethod
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

class GeometricSchedule(nn.Module, Schedule):
    def __init__(self, min: float, max: float):
        self.min = torch.Tensor([min])
        self.max = torch.Tensor([max])
    
    def at(self, t: Tensor):
        return torch.exp(-self.min ** (1 - t) * self.max ** t)

    def derivative_at(self, t):
        return self.at(t) * self.min ** (1-t) * self.max ** t * (self.max.log() - self.min.log())