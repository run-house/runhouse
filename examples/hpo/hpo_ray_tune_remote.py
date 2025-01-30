# # Ray Hyperparameter Tuning with Runhouse
# In this example, we show you how to start a basic hyperparameter tuning using Ray Tune on remote compute.
# You simply need to write your Ray Tune program as you would normally, and then send it to the remote cluster using Runhouse.
# Runhouse handles all the complexities of launching and setting up the remote Ray cluster for you.
import time
from typing import Any, Dict

import runhouse as rh
from ray import tune

# ## Define a Ray Tune program
# We define a Trainable class and a find_minumum function to demonstrate a basic example of using Ray Tune for hyperparameter optimization.
# You should simply think of this as "any regular Ray Tune program" that you would write entirely agnostic of Runhouse.
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


# ## Launch Hyperparameter Tuning using Runhouse
# We will now launch the compute using Runhouse, set up the Ray Cluster, and run the hyperparameter optimization
# on the remote compute.
# * First, we launch 2 nodes of 4 CPUs out of elastic compute on AWS, install the necessary packages via
# the Runhouse image.
# * Then, we send our find_minumum function to the remote cluster with `.to()` and instruct Runhouse to setup Ray with `.distribute("ray")`.
# * Finally, we run the remote function normally as we would locally to start the hyperparameter optimization.

# :::note{.info title="Note"}
# The code to launch, dispatch, and execute should run within a `if __name__ == "__main__":` block, as shown below. Otherwise,
# this script code will run when Runhouse runs the code remotely.
# :::
if __name__ == "__main__":

    num_nodes = 2
    num_cpus_per_node = 4

    img = rh.Image().install_packages(["pyarrow>=9.0.0", "ray[tune]>=2.38.0"])

    cpus = rh.compute(
        name="rh-cpu",
        num_nodes=num_nodes,
        image=img,
        num_cpus=num_cpus_per_node,  # You have other options such as to specify memory and disk size
        gpus=None,  # This example does not need GPUs, but you can specify GPUs like "A100:2" here to get 2 A100 GPUs per node
        provider="aws",  # gcp, kubernetes, etc.
    ).up_if_not()

    remote_find_minimum = rh.function(find_minimum).to(cpus).distribute("ray")
    best_result = remote_find_minimum(num_samples=8)
