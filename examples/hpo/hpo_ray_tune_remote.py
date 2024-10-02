from typing import Any, Dict

import runhouse as rh
from ray import train, tune

NUM_WORKERS = 8
NUM_JOBS = 30


def train_fn(step, width, height):
    return (0.1 + width * step / 100) ** (-1) + height * 0.1


class Trainable(tune.Trainable):
    def setup(self, config: Dict[str, Any]):
        self.reset_config(config)

    def reset_config(self, new_config: Dict[str, Any]):
        self._config = new_config
        return True

    def step(self):
        score = train_fn(**self._config)
        return {"score": score}

    def cleanup(self):
        super().cleanup()

    def load_checkpoint(self, checkpoint_dir: str):
        return None

    def save_checkpoint(self, checkpoint_dir: str):
        return None


def find_minimum(max_concurrent_trials=2, num_samples=4):
    param_space = {
        "step": 100,
        "width": tune.uniform(0, 20),
        "height": tune.uniform(-100, 100),
    }

    tuner = tune.Tuner(
        Trainable,
        tune_config=tune.TuneConfig(
            metric="score",
            mode="min",
            num_samples=num_samples,
            max_concurrent_trials=max_concurrent_trials,
        ),
        run_config=train.RunConfig(stop={"training_iteration": 20}, verbose=2),
        param_space=param_space,
    )
    tuner.fit()
    return tuner.get_results().get_best_result().metrics


if __name__ == "__main__":
    cluster = rh.cluster(
        name="rh-cpu",
        default_env=rh.env(reqs=["ray[tune]"]),
        instance_type="CPU:16+",
        provider="aws",
    ).up_if_not()
    remote_find_minimum = rh.function(find_minimum).to(cluster).distribute("ray")
    best_result = remote_find_minimum()
    print(best_result)
