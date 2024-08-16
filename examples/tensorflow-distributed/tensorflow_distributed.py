# # TensorFlow Multi-node Distributed Training
# A basic example showing how to use Runhouse to Pythonically run a TensorFlow distributed training script on a
# cluster of GPUs. We use the TF_CONFIG environment variable to set up the distributed training environment, and
# create a separate worker (env) for each rank. We then call the replicas concurrently to trigger coordinated
# multi-node training. We're using two single-GPU instances (and therefore two ranks) with the
# MultiWorkerMirroredStrategy, but this same strategy could be used for other TensorFlow distributed strategies.
#
# Despite it being common to use a launcher script to start distributed training, this approach is more flexible and
# allows for more complex orchestration, such as running multiple training jobs concurrently, handling exceptions,
# running distributed training alongside other tasks on the same cluster. It's also significantly easier to debug
# and monitor, as you can see the output of each rank in real-time and get stack traces if a worker fails.

import asyncio
import json
import os

import runhouse as rh
import tensorflow as tf


# ## Define the TensorFlow distributed training logic
# This is the function that will be run on each worker. It initializes the distributed training environment,
# creates a simple model and optimizer, and runs a training loop.
def train_process():
    # Initialize the distributed training environment,
    # per https://www.tensorflow.org/tutorials/distribute/multi_worker_with_keras
    tf_config = json.loads(os.environ["TF_CONFIG"])
    strategy = tf.distribute.MultiWorkerMirroredStrategy()
    num_workers = strategy.num_replicas_in_sync
    print(f"Worker {tf_config['task']['index']} of {num_workers} initialized")

    # Create a simple model and optimizer
    model = tf.keras.Sequential([tf.keras.layers.Dense(10, activation="relu")])
    optimizer = tf.keras.optimizers.SGD(0.01)

    with strategy.scope():
        model.compile(optimizer=optimizer, loss="mse")

    model.fit(
        tf.data.Dataset.from_tensor_slices(
            (tf.random.normal([1000, 10]), tf.random.normal([1000, 1]))
        ).batch(32)
    )

    print(f"Worker {tf_config['task']['index']} finished")


async def train():
    # Create a cluster of 2 GPUs
    gpus_per_node = 1
    num_nodes = 2
    cluster = rh.cluster(
        name=f"rh-{num_nodes}x{gpus_per_node}GPU",
        instance_type=f"A10G:{gpus_per_node}",
        num_instances=num_nodes,
    ).up_if_not()
    train_workers = []
    tf_config = {
        "cluster": {
            "worker": [f"{ip}:12345" for ip in cluster.internal_ips],
        },
    }
    for i in range(num_nodes):
        for j in range(gpus_per_node):
            # Per https://www.tensorflow.org/api_docs/python/tf/distribute/Strategy#attributes
            tf_config["task"] = {"type": "worker", "index": i * gpus_per_node + j}
            env = rh.env(
                name=f"tf_env_{i}",
                reqs=["tensorflow"],
                env_vars={"TF_CONFIG": json.dumps(tf_config)},
                compute={"node_idx": i},
            )
            # While iterating, you can kill the worker processes to stop any pending or hanging calls
            # cluster.delete(env.name)
            train_worker = rh.function(train_process).to(
                cluster, env=env, name=f"train_{i}"
            )
            train_workers.append(train_worker)

    # Call the workers async to run them concurrently (each will connect and wait for the others)
    await asyncio.gather(
        *[train_worker(run_async=True) for train_worker in train_workers]
    )


# :::note{.info title="Note"}
# Make sure that your code runs within a `if __name__ == "__main__":` block, as shown below. Otherwise,
# the script code will run when Runhouse attempts to run code remotely.
# :::
if __name__ == "__main__":
    asyncio.run(train())
