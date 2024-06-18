# ML Infrastructure Design Patterns

This directory describes a set of design patterns and annotated implementations for architecting machine learning
systems with Runhouse based on common workflows. These patterns are based on the experience of
both our team directly, our users' successes with Runhouse, and the architectures we've seen talking to hundreds of
companies doing ML, large and small. They're designed to help you build reliable, scalable, cost-efficient, and
maintainable ML systems.

These patterns are designed to give you intuition and guide your thinking about how to utilize Runhouse.
We've organized them based on the high-level workflows in which they're most likely to be found, but many of them
are general-purpose and can be used in multiple contexts. We've also tried to include some context about what
alternatives you might use today to achieve the same goals, and why you might choose one approach over another. We
do not include example code for the alternatives, but we're happy to help you think through architectures and
options in the [Runhouse discord](https://discord.gg/RnhB6589Hs).

# 📚 Patterns

We've organized the patterns into the following categories:
 * [Optimizing I/O] - Strategies for synchronizing data communication with computation to minimize latency
 * [Caching & State] - Strategies for utilizing memory and state to minimize data movement and latency
 * [Scaling up] - Strategies for vertically scaling, or using more powerful machines
 * [Scaling out] - Strategies for horizontally scaling, or using more machines
 * [Heterogeneity] - Strategies for utilizing multiple types of compute, regions, clouds, or clusters
 * [Lowering] - Strategies for using optimized vectorization, compilation, or distributed libraries
 * [Fault-tolerance] - Strategies for handling failures
 * [Scaling across] - Strategies for scaling across multiple workflows and jobs
 * [Operationalization] - Strategies for managing the lifecycle of ML systems and collaborating on them

## Data Loading & Processing

Offline, batch processing is a common first step in many ML workflows. This is where you load data, preprocess it, and
write it back out to disk or a database. This is often the most time-consuming part of the workflow, and there are many
ways to optimize it.
 * [Optimizing I/O] Concurrent data loading/writes - multithreading and async
 * [Caching & State] Processing in-place / avoiding data roundtrips - filesystem, pinning, pipelining
 * [Scaling up] Parallel data processing - multiprocessing
 * [Scaling out] Parallel data processing  - Multi-node
 * [Heterogeneity] Parallel data processing - Multi-cluster, spot, multi-cloud, or multi-region
 * [Lowering] Using vectorized and distributed processing libraries - ray.data, Spark, Dask
 * [All together] Batch inference - parallel, distributed, and heterogeneous (BERT embeddings)
 * [All together] Concurrency, vertical, horizontal, and heterogeneous all together

## Training & Evaluation

Training and evaluation are often where the lion's share of iteration and debugging happen, and where much of the
struggle arises graduating the ML methods from scripts to living systems. Flexibility, speed, and reliability are key.
 * [Scaling up] Distributed training - PyTorch, TensorFlow, Horovod
 * [Scaling out] Multi-node training - ray.train(), Hugging Face Accelerate (DeepSpeed)
 * [Optimizing I/O] Pipelining and streaming preprocessed batches
 * [Caching & State] Avoiding data reloading - caching preprocessed batches (training and eval)
 * [Scaling across] Factoring out shared services - Evaluation
 * [Fault-tolerance] OOM, failure, or preemption
 * [Heterogeneity] Greedy try-and-fail training - mixing compute types or model architectures
 * [Scaling out] Hyperparameter Optimization
 * [All together] Online learning pipeline - training, evaluation, and deployment
 * [Operationalization] Research-to-Airflow and back

## Serving

Serving is where the rubber meets the road in ML systems. It's where you take your mostly latency insensitive workloads
and make them real-time, low-latency, and cost-efficient. It's also where you have to think about how to manage the
lifecycle of your models, how to scale them, and how to monitor them.
 * [Caching & State] Avoiding model reloading / GPU communications
 * [Scaling across] Factoring out shared services - FastAPI example
 * [Lowering] Using compiled, batched, and distributed engines - vLLM, TorchServe, Triton, TensorFlow Serving
 * [Scaling out] Round-robin, rate-limiting, and spilling over to other services
 * [Scaling across] Blue/Green deployment
 * [All together] Real-time inference - low-latency, high-throughput, and cost-efficient (Langchain App)
 * [Operationalization] Research-to-Kubernetes and back

# Contributing

As you can see, we're still building out this directory. If you have a pattern you'd like to see, or would like to
contribute, please let us know in the [Runhouse discord](https://discord.gg/RnhB6589Hs) or by opening an issue or PR.
Successful contributions will be rewarded with Runhouse swag!
