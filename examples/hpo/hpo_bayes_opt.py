import time

import runhouse as rh

NUM_WORKERS = 8
NUM_JOBS = 30


def train_fn(step, width, height):
    time.sleep(5)
    return (0.1 + width * step / 100) ** (-1) + height * 0.1


if __name__ == "__main__":
    from bayes_opt import BayesianOptimization

    cluster = rh.cluster(
        name="rh-4x16-cpu", instance_type="CPU:16", num_nodes=4, provider="aws"
    ).up_if_not()
    train_env = rh.env(name="worker_env", compute={"CPU": 8})
    train_fn = rh.function(train_fn).to(cluster, env=train_env)
    train_fn_pool = train_fn.distribute("queue", replicas=NUM_WORKERS)

    optimizer = BayesianOptimization(
        f=train_fn_pool,
        pbounds={"width": (0, 20), "height": (-100, 100)},
        verbose=2,
        random_state=1,
    )
    optimizer.maximize(init_points=NUM_WORKERS, n_iter=NUM_JOBS)
    print(f"Optimization finished. Best parameters found: {optimizer.max}")
