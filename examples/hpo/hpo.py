import asyncio
import time

import numpy as np

import runhouse as rh

NUM_WORKERS = 8
NUM_JOBS = 30


def train_fn(step, width, height):
    time.sleep(5)
    return (0.1 + width * step / 100) ** (-1) + height * 0.1


def generate_params():
    return {"width": np.random.uniform(0, 1), "height": np.random.uniform(0, 1)}


async def find_best_params():
    cluster = rh.cluster(
        name="rh-4x16-cpu", instance_type="CPU:16", num_instances=4, provider="aws"
    ).up_if_not()
    train_env = rh.env(name="worker_env", compute={"CPU": 8})
    remote_train_fn = rh.function(train_fn).to(cluster, env=train_env)
    available_worker_fns = [remote_train_fn] + remote_train_fn.replicate(
        NUM_WORKERS - 1
    )

    async def run_job(step):
        while not available_worker_fns:
            await asyncio.sleep(1)
        worker_fn = available_worker_fns.pop(0)
        next_point_to_probe = generate_params()

        print(f"Calling step {step} on point {next_point_to_probe}")
        target = await worker_fn(step=step, **next_point_to_probe, run_async=True)
        print(f"Returned step {step} with value {target}")

        available_worker_fns.append(worker_fn)
        return next_point_to_probe, target

    results = await asyncio.gather(
        *[run_job(counter) for counter in range(NUM_JOBS)], return_exceptions=True
    )

    max_result = max(results, key=lambda x: x[1])
    print(f"Optimization finished. Best parameters found: {max_result}")


if __name__ == "__main__":
    asyncio.run(find_best_params())
