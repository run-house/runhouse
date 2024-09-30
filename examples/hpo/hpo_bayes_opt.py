from functools import partial

import runhouse as rh

from bayes_opt import BayesianOptimization

NUM_WORKERS = 8
NUM_JOBS = 30


def train_fn(x, y):
    return -(x**2) - (y - 1) ** 2 + 1


if __name__ == "__main__":
    cluster = (
        rh.cluster(
            name="rh-4x16-cpu",
            instance_type="CPU:16",
            num_nodes=2,
            provider="aws",
            default_env=rh.env(
                reqs=["bayesian-optimization"], env_vars={"RH_LOG_LEVEL": "INFO"}
            ),
        )
        .save()
        .up_if_not()
    )
    train_env = rh.env(name="worker_env", load_from_den=False)
    remote_train_fn = rh.function(train_fn).to(cluster, env=train_env)
    train_fn_pool = remote_train_fn.distribute(
        "queue", num_replicas=NUM_WORKERS, replicas_per_node=NUM_WORKERS // 2
    )

    optimizer = BayesianOptimization(
        f=partial(train_fn_pool, stream_logs=False),
        pbounds={"x": (-2, 2), "y": (-3, 3)},
        verbose=2,
        random_state=1,
    )
    optimizer.maximize(init_points=NUM_WORKERS, n_iter=NUM_JOBS)
    print(f"Optimization finished. Best parameters found: {optimizer.max}")
