# PyTorch Multi-node Distributed Training

A basic example showing how to use Runhouse to Pythonically run a PyTorch distributed training script on a
cluster of GPUs. Often distributed training is launched from multiple parallel CLI commands
(`python -m torch.distributed.launch ...`), each spawning separate training processes (ranks).
Here, we're creating each process as a separate worker (`env`) on the cluster, sending our training function
into each worker, and calling the replicas concurrently to trigger coordinated multi-node training
(`torch.distributed.init_process_group` causes each to wait for all to connect, and sets up the distributed
communication). We're using two single-GPU instances (and therefore two ranks) for simplicity, but we've included
the basic logic to handle multi-GPU nodes as well, where you'd add more worker processes per node and set `device_ids`
accordingly.

Despite it being common to use a launcher script to start distributed training, this approach is more flexible and
allows for more complex orchestration, such as running multiple training jobs concurrently, handling exceptions,
running distributed training alongside other tasks on the same cluster. It's also significantly easier to debug
and monitor, as you can see the output of each rank in real-time and get stack traces if a worker fails.
