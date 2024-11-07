import asyncio
import time

import runhouse as rh

NUM_WORKERS = 8
NUM_JOBS = 30


def train_fn(step, width, height):
    time.sleep(5)
    return (0.1 + width * step / 100) ** (-1) + height * 0.1


async def find_best_params():
    from bayes_opt import BayesianOptimization, UtilityFunction

    cluster = rh.cluster(
        name="rh-4x16-cpu", instance_type="CPU:16", num_nodes=4, provider="aws"
    ).up_if_not()
    train_env = rh.env(name="worker_env", compute={"CPU": 8})
    remote_fn = rh.function(train_fn).to(cluster, env=train_env)
    worker_fns = [remote_fn] + remote_fn.replicate(NUM_WORKERS - 1)

    optimizer = BayesianOptimization(
        f=None,
        pbounds={"width": (0, 20), "height": (-100, 100)},
        verbose=2,
        random_state=1,
    )
    utility = UtilityFunction(kind="ucb", kappa=2.5, xi=0.0)

    async def run_job(step):
        while not worker_fns:
            await asyncio.sleep(1)
        worker_fn = worker_fns.pop(0)
        hyperparams = optimizer.suggest(utility)

        print(f"Calling step {step} on point {hyperparams}")
        target = await worker_fn(step=step, **hyperparams, run_async=True)
        print(f"Returned step {step} with value {target}")

        optimizer.register(hyperparams, target)
        utility.update_params()

        worker_fns.append(worker_fn)

    futs = [run_job(counter) for counter in range(NUM_JOBS)]
    await asyncio.gather(*futs, return_exceptions=True)

    print(f"Optimization finished. Best parameters found: {optimizer.max}")


if __name__ == "__main__":
    asyncio.run(find_best_params())
