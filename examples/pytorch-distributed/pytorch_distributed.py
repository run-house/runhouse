# # PyTorch Multi-node Distributed Training
# A basic example showing how to use Runhouse to Pythonically run a PyTorch distributed training script on a
# cluster of GPUs. Often distributed training is launched from multiple parallel CLI commands
# (`python -m torch.distributed.launch ...`), each spawning separate training processes (ranks).
# Here, we're creating each process as a separate worker (`env`) on the cluster, sending our training function
# into each worker, and calling the replicas concurrently to trigger coordinated multi-node training
# (`torch.distributed.init_process_group` causes each to wait for all to connect, and sets up the distributed
# communication). We're using two single-GPU instances (and therefore two ranks) for simplicity, but we've included
# the basic logic to handle multi-GPU nodes as well, where you'd add more worker processes per node and set `device_ids`
# accordingly.
#
# Despite it being common to use a launcher script to start distributed training, this approach is more flexible and
# allows for more complex orchestration, such as running multiple training jobs concurrently, handling exceptions,
# running distributed training alongside other tasks on the same cluster. It's also significantly easier to debug
# and monitor, as you can see the output of each rank in real-time and get stack traces if a worker fails.

import asyncio

import runhouse as rh
import torch

from torch.nn.parallel import DistributedDataParallel as DDP


# ## Define the PyTorch distributed training logic
# This is the function that will be run on each worker. It initializes the distributed training environment,
# creates a simple model and optimizer, and runs a training loop.
def train_process():
    # Initialize the distributed training environment,
    # per https://pytorch.org/docs/stable/distributed.html#initialization
    torch.distributed.init_process_group(backend="nccl")
    rank = torch.distributed.get_rank()
    print(f"Rank {rank} of {torch.distributed.get_world_size()} initialized")

    # Create a simple model and optimizer
    device_id = rank % torch.cuda.device_count()
    model = torch.nn.Linear(10, 1).cuda(device_id)
    model = DDP(model, device_ids=[device_id])
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    # Perform a simple training loop
    for epoch in range(10):
        optimizer.zero_grad()
        output = model(torch.randn(10).cuda())
        loss = output.sum()
        loss.backward()
        optimizer.step()

        print(f"Rank {rank}: Epoch {epoch}, Loss {loss.item()}")

    # Clean up the distributed training environment
    torch.distributed.destroy_process_group()


async def train():
    # Create a cluster of 2 GPUs
    gpus_per_node = 1
    num_nodes = 2
    cluster = rh.cluster(
        name=f"rh-{num_nodes}x{gpus_per_node}GPU",
        instance_type=f"A10G:{gpus_per_node}",
        num_nodes=num_nodes,
    ).up_if_not()
    train_workers = []
    for i in range(num_nodes):
        for j in range(gpus_per_node):
            # Per https://pytorch.org/docs/stable/distributed.html#environment-variable-initialization
            dist_config = {
                "MASTER_ADDR": cluster.internal_ips[0],
                "MASTER_PORT": "12345",
                "RANK": str(i * gpus_per_node + j),
                "WORLD_SIZE": str(num_nodes * gpus_per_node),
            }
            env = rh.env(
                name=f"pytorch_env_{i}",
                reqs=["torch"],
                env_vars=dist_config,
                compute={"node_idx": i},
            )
            # While iterating, you can kill the worker processes to stop any pending or hanging calls
            # cluster.delete(env.name)
            train_worker = rh.function(train_process).to(
                cluster, env=env, name=f"train_{i}"
            )
            train_workers.append(train_worker)

    # Call the workers concurrently with asyncio (each will connect and
    # wait for the others during init_process_group)
    # Note: Because this is regular async, we'll get errors and stack traces if a worker fails
    await asyncio.gather(
        *[train_worker(run_async=True) for train_worker in train_workers]
    )


# :::note{.info title="Note"}
# Make sure that your code runs within a `if __name__ == "__main__":` block, as shown below. Otherwise,
# the script code will run when Runhouse attempts to run code remotely.
# :::
if __name__ == "__main__":
    asyncio.run(train())
