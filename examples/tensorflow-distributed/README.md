# TensorFlow Multi-node Distributed Training

A basic example showing how to use Runhouse to Pythonically run a TensorFlow distributed training script on a
cluster of GPUs. We use the `TF_CONFIG` environment variable to set up the distributed training environment, and
create a separate worker (env) for each rank. We then call the replicas concurrently to trigger coordinated
multi-node training. We're using two single-GPU instances (and therefore two ranks) with the
MultiWorkerMirroredStrategy, but this same strategy could be used for other TensorFlow distributed strategies.

Despite it being common to use a launcher script to start distributed training, this approach is more flexible and
allows for more complex orchestration, such as running multiple training jobs concurrently, handling exceptions,
running distributed training alongside other tasks on the same cluster. It's also significantly easier to debug
and monitor, as you can see the output of each rank in real-time and get stack traces if a worker fails.
