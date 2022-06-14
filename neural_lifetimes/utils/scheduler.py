from abc import ABC, abstractmethod


class WeightScheduler(ABC):
    @property
    @abstractmethod
    def weight(self) -> float:
        pass

    @abstractmethod
    def step(self) -> None:
        pass


class LinearWarmupScheduler(WeightScheduler):
    def __init__(self, n_cold_steps: int, n_warmup_steps: int, target_weight: float) -> None:
        super().__init__()
        self.n_cold_steps = n_cold_steps
        self.n_warmup_steps = n_warmup_steps
        self.target_weight = target_weight
        self._step = 1

    @property
    def weight(self) -> float:
        if self._step < self.n_cold_steps:
            return 0.0
        elif self._step < self.n_warmup_steps + self.n_cold_steps:
            return (self._step - self.n_cold_steps) / self.n_warmup_steps * self.target_weight
        else:
            return self.target_weight

    def step(self):
        self._step += 1


class ExponentialWarmupScheduler(WeightScheduler):
    pass
