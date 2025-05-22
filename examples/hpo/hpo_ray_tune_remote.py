# # Ray Hyperparameter Tuning
# In this example, we show you how to start a basic hyperparameter tuning using Ray Tune on remote compute.
# You simply need to write your Ray Tune program as you would normally, and then send it to the remote cluster.
# Kubetorch handles all the complexities of launching and setting up the remote Ray cluster for you.
import time
from typing import Any, Dict

import kubetorch as kt
from ray import tune

# ## Define a Ray Tune program
# We define a Trainable class and a find_minumum function to demonstrate a basic example of using Ray Tune for hyperparameter optimization.
# You should simply think of this as "any regular Ray Tune program" that you would write entirely agnostic of Kubetorch.
# * Train_fn is a dummy training function that takes in a step number, width, and height as arguments and returns a score.
# * The Trainable class is a subclass of tune.Trainable that implements the training logic.
# * The find_minimum function sets up the hyperparameter search space and launches the hyperparameter optimization using Ray Tune.


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


def find_minimum(num_concurrent_trials=None, num_samples=1, metric_name="score"):
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


# ## Launch Hyperparameter Tuning
# We will now dispatch the program, set up Ray, and run the hyperparameter optimization
# on the remote compute.
if __name__ == "__main__":
    head = kt.Compute(num_cpus=4, image=kt.Image(image_id="rayproject/ray")).distribute(
        "ray", num_nodes=4
    )
    remote_find_minimum = kt.fn(find_minimum).to(head)
    best_result = remote_find_minimum(num_samples=8)
