from functools import partial

import runhouse as rh

from bayes_opt import BayesianOptimization

NUM_WORKERS = 8
NUM_JOBS = 30


@rh.deploy(
    num_cpus="4+",
    num_nodes=2,
    provider="kubernetes",
    image=rh.Image(name="hpo").install_packages(["bayesian-optimization"]),
    # image=rh.images.Ubuntu.install_packages(["bayesian-optimization"]),
)
@rh.distribute("pool", num_replicas=NUM_WORKERS)
def train_fn(x, y):
    return -(x**2) - (y - 1) ** 2 + 1


if __name__ == "__main__":
    optimizer = BayesianOptimization(
        f=partial(train_fn, stream_logs=False),
        pbounds={"x": (-2, 2), "y": (-3, 3)},
        verbose=2,
        random_state=1,
    )
    optimizer.maximize(init_points=NUM_WORKERS, n_iter=NUM_JOBS)
    print(f"Optimization finished. Best parameters found: {optimizer.max}")
