from functools import partial

import runhouse as rh

from bayes_opt import BayesianOptimization

NUM_WORKERS = 8
NUM_JOBS = 30


def train_fn(x, y):
    return -(x**2) - (y - 1) ** 2 + 1


if __name__ == "__main__":
    cluster = rh.cluster(
        name="rh-4x16-cpu",
        instance_type="CPU:4+",
        num_nodes=2,
        provider="kubernetes",
        default_env=rh.env(
            reqs=["bayesian-optimization"],
        ),
    ).up_if_not()
    remote_train_fn = rh.function(train_fn).to(cluster)
    train_fn_pool = remote_train_fn.distribute(
        "pool", num_replicas=NUM_WORKERS, replicas_per_node=NUM_WORKERS // 2
    )

    optimizer = BayesianOptimization(
        f=partial(train_fn_pool, stream_logs=False),
        pbounds={"x": (-2, 2), "y": (-3, 3)},
        verbose=2,
        random_state=1,
    )
    optimizer.maximize(init_points=NUM_WORKERS, n_iter=NUM_JOBS)
    print(f"Optimization finished. Best parameters found: {optimizer.max}")
