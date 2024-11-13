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
import runhouse as rh
from bayes_opt import BayesianOptimization

from ..pytorch.pytorch_distributed import train_loop

# :::note{.info title="Note"}
# Make sure that your code runs within a `if __name__ == "__main__":` block, as shown below. Otherwise,
# the script code will run when Runhouse attempts to run code remotely.
# :::
if __name__ == "__main__":
    gpus_per_node = 1
    gpus_per_worker = 2
    num_hpo_workers = 2
    cluster = rh.cluster(
        name=f"rh-{num_nodes}x{gpus_per_node}GPU",
        instance_type=f"A10G:{gpus_per_node}",
        num_nodes=gpus_per_worker * num_hpo_workers,
    ).up_if_not()
    train_ddp = (
        rh.function(train_loop)
        .to(cluster)
        .distribute(
            "pytorch",
            replicas_per_node=gpus_per_node,
            replicas=gpus_per_node * gpus_per_worker,
        )
    )
    train_ddp_pool = train_ddp.distribute("pool", replicas=num_hpo_workers)

    optimizer = BayesianOptimization(
        f=train_ddp_pool,
        pbounds={"width": (0, 20), "height": (-100, 100)},
        verbose=2,
        random_state=1,
    )
    optimizer.maximize(init_points=num_hpo_workers, n_iter=10)
    print(f"Optimization finished. Best parameters found: {optimizer.max}")
