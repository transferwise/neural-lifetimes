from neural_lifetimes.utils.scheduler import WeightScheduler, LinearWarmupScheduler, ExponentialWarmupScheduler


class Test_LinearWarmupScheduler:
    @staticmethod
    def scheduler(n_cold_steps: int = 2, n_warmup_steps: int = 5, target_weight: float = 2):
        return LinearWarmupScheduler(
            n_cold_steps=n_cold_steps,
            n_warmup_steps=n_warmup_steps,
            target_weight=target_weight,
        )

    def test_constructor(self):
        self.scheduler()

    def test_scheduling(self):
        scheduler = self.scheduler()

        scheduler_weights = []
        for i in range(10):
            scheduler_weights.append(scheduler.weight)
            scheduler.step()

        expected_weights = [0, 0, 0.4, 0.8, 1.2, 1.6, 2, 2, 2, 2]
        assert all([s == e for s, e in zip(scheduler_weights, expected_weights)])
