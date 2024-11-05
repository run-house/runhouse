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
import os

import runhouse as rh
import torch

from torch.nn.parallel import DistributedDataParallel as DDP


# ## Define the PyTorch distributed training logic
# This is the function that will be run on each worker. It initializes the distributed training environment,
# creates a simple model and optimizer, and runs a training loop.
def train_loop(epochs):
    # Initialize the distributed training environment,
    # per https://pytorch.org/docs/stable/distributed.html#initialization
    print("Initializing distributed training environment")
    print(os.environ["MASTER_ADDR"])
    print(os.environ["MASTER_PORT"])
    print(os.environ["RANK"])
    print(os.environ["WORLD_SIZE"])
    torch.distributed.init_process_group(backend="nccl")
    print(f"Rank {torch.distributed.get_rank()} initialized")
    rank = torch.distributed.get_rank()
    print(f"Rank {rank} of {torch.distributed.get_world_size()} initialized")

    # Create a simple model and optimizer
    device_id = rank % torch.cuda.device_count()
    model = torch.nn.Linear(10, 1).cuda(device_id)
    model = DDP(model, device_ids=[device_id])
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    # Perform a simple training loop
    for epoch in range(epochs):
        optimizer.zero_grad()
        output = model(torch.randn(10).cuda())
        loss = output.sum()
        loss.backward()
        optimizer.step()

        print(f"Rank {rank}: Epoch {epoch}, Loss {loss.item()}")

    # Clean up the distributed training environment
    torch.distributed.destroy_process_group()


# :::note{.info title="Note"}
# Make sure that your code runs within a `if __name__ == "__main__":` block, as shown below. Otherwise,
# the script code will run when Runhouse attempts to run code remotely.
# :::
if __name__ == "__main__":
    gpus_per_node = 1
    num_nodes = 2
    cluster = (
        rh.cluster(
            name=f"rh-{num_nodes}x{gpus_per_node}GPU",
            instance_type=f"A10G:{gpus_per_node}",
            num_instances=num_nodes,
            default_env=rh.env(reqs=["torch"]),
        )
        .save()
        .up_if_not()
    )
    # cluster.restart_server()
    remote_train_loop = rh.function(train_loop).to(cluster)
    train_ddp = remote_train_loop.distribute(
        "pytorch",
        num_replicas=num_nodes * gpus_per_node,
        replicas_per_node=gpus_per_node,
    )
    train_ddp(epochs=10)
