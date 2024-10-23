Using Runhouse with Common ML Tools and Libraries
==========================================

Runhouse is built to be extremely unopinionated and works closely with familiar tools and libraries in the ML ecosystem.
Its APIs are similar to those of other systems, making it easy to compare and contrast. In many cases, Runhouse complements or extends existing tools,
while in other instances, it can replace them entirely.

Notebooks and IDEs (Hosted or Local)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
ML engineers often lose the ability to develop locally when dataset sizes and the need for accelerated compute exceed the capabilities of local hardware.
Hosted notebooks have become a common solution for rapid iteration during the research phase, but they can fragment the development workflow and introduce
a "research-to-production" gap that doesn't exist in traditional software engineering.

Runhouse advocates for defining ML programs and pipelines using standard Python code. Classes and functions should be portable,
importable, testable, and managed with software best practices within a team repository. Code development is best done in a traditional IDE,
but Runhouse is flexible in how the code is executed interactively. You can launch compute, dispatch code,
and execute it in script form or use Python notebooks as interactive shells.

With Runhouse, classes are remote objects that can be accessed through multi-threaded calls.
If we instantiate a remote trainer class and launch training loops in one local thread, we can
also establish a separate connection to the remote object in another thread. This allows us to perform multiple tasks
simultaneously with the same remote object: for example, running training epochs, saving model checkpoints, and conducting test evaluations.
These tasks can be done from three async calls, three scripts, or three notebook cells.

We show here how a LoRA Fine Tuner class can be launched from a notebook
in `this example <https://github.com/run-house/runhouse/tree/1b047c9b22839c212a1e2674407959e7e775f21b/examples/lora-example-with-notebook>`_.

Workflow orchestrators (e.g. Airflow, Prefect, Dagster, Flyte, Metaflow, Argo)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Workflow orchestrators are excellent for monitoring, telemetry, fault tolerance, and scheduling, so we recommend using them for these tasks.
However, they shouldn't act as your application or runtime. For example, instead of converting a Python application into an Airflow DAG to run
training on a GPU, you simply make an HTTP call within an Airflow step to trigger a training function running as a service. ML development
with "notebooks-plus-DAGs" leads to poor reproducibility, bad debuggability, and slow research-to-production.

.. image:: https://runhouse-tutorials.s3.amazonaws.com/R2P+WO+Runhouse.jpg
  :alt: Fragmented research and production in separate compute environments
  :width: 650

By avoiding the need to repack ML code into pipelines, teams can significantly reduce research-to-production time and improve debuggability.
Runhouse ensures that the code committed to the team repository will execute reproducibly in production without additional translation.
Additionally, iteration loops remain fast in production, whether for debugging or further development.
ML engineers can reproduce a failed production run locally by copying the dispatch code, quickly debug and iterate, then push the changes.
This approach is much faster than the traditional 20+ minute cycles required to rebuild and rerun orchestrator pipelines.

.. image:: https://runhouse-tutorials.s3.amazonaws.com/R2P+W+Runhouse.jpg
  :alt: Unified dispatch from notebooks and nodes with Runhouse
  :width: 650

There are many clever patterns that Runhouse enables in conjunction with orchestrators that save time and money.

* Reusing of the same compute across multiple tasks while separating the steps in the orchestrator for clarity. For instance, avoiding the I/O overhead of repeatedly writing/reading data for each step of an Argo/Kubeflow pipeline.
* Sharing a single service to be shared across multiple orchestrator pipelines. For instance, a single embeddings service can be used by multiple pipelines.
* Maintaining a single orchestrator, but dispatching each pipeline step to arbitrary clusters, regions, or even clouds. For instance, do pre-processing on AWS, but GPU training on GCP where you have quota/credits.
* Catching and handling errors natively from the orchestrator node, since the orchestrator runtime is a Python-based driver for the execution. For instance, on fail due to OOM, launch a larger box and rerun.

Distributed frameworks (e.g. Ray, Spark, Elixr)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Runhouse is a perfect complement to distributed frameworks, letting you use these frameworks in a less disruptive way.

Distributed frameworks are built to offload execution to different processes or nodes *within* their own cluster environments.
Runhouse is focused on dispatching execution to compute resources *outside* Runhouse's own runtime (which is Python)
and coordinating execution across different types of clusters.
As an example, when using Ray with Runhouse, you use Runhouse to launch a cluster and then send a function to the head node of a Ray cluster, where Ray will execute it as usual.

This approach fixes some sharp edges of traditional distributed frameworks. First, because the local
and remote compute environments are decoupled, so there is no shared runtime
that could fail if one part disconnects or experiences downtime, whereas without Runhouse, an out-of-memory
error in a node has a high chance of crashing the entire application. Runhouse also enables the use of multiple clusters in a single application,
and also supports sharing a cluster across multiple different callers.

.. image:: https://runhouse-tutorials.s3.amazonaws.com/Runhouse+and+Distributed+DSLs.jpg
  :alt: Runhouse distributes from Python to a Ray Cluster (or Spark)
  :width: 650

Serverless frameworks (e.g. AWS Lambda, Google Cloud Functions, Fireworks, Modal)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Serverless frameworks enable on-the-fly service allocation, and similarly to Runhouse, abstract compute management away from engineers.
However, they often require pre-packaging or command-line interface (CLI) launches outside of
standard Python environments. Runhouse, on the other hand, runs entirely within a Python interpreter, allowing it to extend the
compute capabilities of existing Python applications. Very critically, Runhouse lets you **allocate resources within your own infrastructure**.

Serverless solutions are a broad category, and many serverless solutions aren't suitable for ML workloads. For instance, AWS Lambda struggles with large datasets, GPU-accelerated tasks,
or long-running jobs. Runhouse can offload these tasks to ephemerally launched, but powerful compute that lasts until the job is done.
Even when evaluating serverless solutions optimized for ML, it's essential to distinguish between those optimized for inference and Runhouse.
For inference, you likely prioritize latency, cold start times and typically execute on a few limited types of hardware.
But if you are considering executing recurring training, for instance, Runhouse is significantly more optimized; you have better hardware heterogeneity,
debuggability, statefulness across epochs, and the ability to efficiently use compute.

Slurm-Style Compute Interfaces (e.g. Slurm, SkyPilot, Mosaic, SageMaker Training)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
In this category of Slurm-style solutions, compute is allocated on the fly and scripts are used as entry points.
For heavyweight jobs that are run manually, such as a research lab training a large language
model over hundreds of GPUs, this style of execution works quite well. However, for recurring enterprise ML use cases, there are several distinct disadvantages
that Runhouse fixes.

* Limited control over execution flow, making it difficult to dispatch multiple stages or function calls to the same compute resource (e.g., loading datasets, training, and evaluation).
* Weak fault tolerance due to the inability to catch and handle remote exceptions (all exception handling must occur within the script, leaving little recourse for issues like out-of-memory errors)
* Configuration sprawl as training scripts branch for each new method or experiment, and combinations of settings that work together grow sparser and sparser.

For elastic compute scenarios, Runhouse uses SkyPilot to allocate resources but goes beyond that by offering (re)deployment and execution management.
This restores control over execution, adds fault tolerance, and allows all compute configurations to be defined in code.
