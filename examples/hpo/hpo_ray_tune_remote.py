import time
from typing import Any, Dict

import runhouse as rh
from ray import train, tune

NUM_WORKERS = 8
NUM_JOBS = 30


def train_fn(step, width, height):
    time.sleep(5)
    return (0.1 + width * step / 100) ** (-1) + height * 0.1


class Trainable(tune.Trainable):
    def setup(self, config: Dict[str, Any]):
        self.step_num = 0
        self.reset_config(config)

    def reset_config(self, new_config: Dict[str, Any]):
        self._config = new_config
        return True

    def step(self):
        score = train_fn(self.step_num, **self._config)
        self.step_num += 1
        return {"score": score}

    def cleanup(self):
        super().cleanup()

    def load_checkpoint(self, checkpoint_dir: str):
        return None

    def save_checkpoint(self, checkpoint_dir: str):
        return None


# Alternative trainable that can be passed to tune.Tuner
def trainable(config):
    for step_num in range(10):
        from hpo_train_fn import train_fn

        score = train_fn(step_num, **config)
        train.report(score=score)


def find_minimum(num_concurrent_trials=2, num_samples=4, metric_name="score"):
    search_space = {
        "width": tune.uniform(0, 20),
        "height": tune.uniform(-100, 100),
    }

    tuner = tune.Tuner(
        Trainable,
        tune_config=tune.TuneConfig(
            metric=metric_name,
            mode="max",
            max_concurrent_trials=num_concurrent_trials,
            num_samples=num_samples,
            reuse_actors=True,
        ),
        param_space=search_space,
    )
    tuner.fit()
    return tuner.get_results().get_best_result()


if __name__ == "__main__":
    cluster = rh.cluster(
        name="rh-cpu",
        image=rh.Image("tune").install_packages(["ray[tune]>==2.38.0"]),
        instance_type="CPU:4+",
        provider="aws",
    ).up_if_not()
    remote_find_minimum = rh.function(find_minimum).to(cluster).distribute("ray")
    best_result = remote_find_minimum()
