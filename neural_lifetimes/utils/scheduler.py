from abc import ABC, abstractmethod


class WeightScheduler(ABC):
    """Abstract baseclass for the weight schedulers.

    Classes of this template require a weight attribute returning the current weight and a step method.
    """

    @property
    @abstractmethod
    def weight(self) -> float:
        pass

    @abstractmethod
    def step(self) -> None:
        pass


class LinearWarmupScheduler(WeightScheduler):
    def __init__(self, n_cold_steps: int, n_warmup_steps: int, target_weight: float) -> None:
        """The LinearWarmupScheduler implements a linear increase in the weight to start up training.

        Often model training benefit from starting training with no penalty and slowing increasing it after a few
        first steps. This is called warmup. In particular, the information bottleneck requires this. The linear
        scheduler returns penalty weight 0 until step ``n_cold_steps`` is reached and then linearly interpolates for
        ``n_warmup_steps`` until the ``target_weight`` is reached.

        Args:
            n_cold_steps (int): Number of steps to conduct with 0 weight.
            n_warmup_steps (int): Number of steps to warmup weight.
            target_weight (float): Target weight.
        """
        super().__init__()
        self.n_cold_steps = n_cold_steps
        self.n_warmup_steps = n_warmup_steps
        self.target_weight = target_weight
        self._step = 1

    @property
    def weight(self) -> float:
        """Get the current weight.

        Returns:
            float: Weight.
        """
        if self._step <= self.n_cold_steps:
            return 0.0
        elif self._step < self.n_warmup_steps + self.n_cold_steps:
            return (self._step - self.n_cold_steps) / self.n_warmup_steps * self.target_weight
        else:
            return self.target_weight

    def step(self):
        """Increases the step."""
        self._step += 1


class ExponentialWarmupScheduler(WeightScheduler):
    def __init__(self, n_cold_steps: int, n_warmup_steps: int, target_weight: float, gamma: float) -> None:
        """The ExpoentialWarmupScheduler implements an exponential increase in the weight to start up training.

        Often model training benefit from starting training with no penalty and slowing increasing it after a few
        first steps. This is called warmup. In particular, the information bottleneck requires this. The exponential
        scheduler returns penalty weight 0 until step ``n_cold_steps`` is reached and then increases the weight every
        step by factor ``gamma`` for ``n_warmup_steps`` steps, when ``target_weight`` is reached.

        Args:
            n_cold_steps (int): Number of cold start steps with 0 weight.
            n_warmup_steps (int): Number of steps to warmup weight.
            target_weight (float): Target weight.
            gamma (float): Factor to increase learning rate.
        """
        super().__init__()
        self.n_cold_steps = n_cold_steps
        self.n_warmup_steps = n_warmup_steps
        self.target_weight = target_weight
        self.gamma = gamma
        self._step = 1

    @property
    def weight(self) -> float:
        """Get the current weight.

        Returns:
            float: Weight.
        """
        if self._step <= self.n_cold_steps:
            return 0.0
        elif self._step < self.n_warmup_steps + self.n_cold_steps:
            return self.target_weight / self.gamma ** (self.n_warmup_steps + self.n_cold_steps - self._step)
        else:
            return self.target_weight

    def step(self):
        """Increases the step."""
        self._step += 1
