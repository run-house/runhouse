Why Runhouse
=====================

Runhouse is a Python library that enables applications to seamlessly utilize remote compute infrastructure by deploying and invoking remote services on demand.
In essence, Runhouse provides "serverless" execution using your own infrastructure.

Runhouse is specifically designed for machine learning workloads — including online and offline tasks, training, and inference —
where the need for heterogeneous remote compute resources is common, and flexibility is essential to keep development cycles fast and costs low.

Key Benefits
------------

Runhouse solves a few major problems for machine learning and AI teams:

#. **Iterability**: Developing with Runhouse feels like working locally, even if the code is executing on powerful, multi-node remote hardware.
   In research, avoid writing non-standard code in hosted notebooks; in production, don't iterate by building and resubmitting pipelines.
   The team writes standard Python code locally, and it takes less than 2 seconds per iteration to redeploy the code to remote compute.
   The remote filesystem and any unaffected remote objects or functions remain accessible across iterations.
#. **Debuggbility**: With Runhouse, there is perfect reproducibility between local and scheduled production execution.
   Research code that works is already production-ready, while any production runs that fail can be debugged locally.
   The combination of identical execution and fast iteration enables a straightforward, rapid debugging loop.
#. **Cost**: Runhouse offers the flexibility to allocate compute resources only when needed, right-size instances based on workload,
   work across multiple regions or clouds for lower costs, and share compute and services across tasks.
   Users typically see cost savings of 50-75%, depending on the workload.
#. **Development at Scale**: Adopting powerful, GPU accelerated hardware or distributed clusters (Spark, Ray) can be
   disruptive. All development, debugging, automation, and deployment to occur on their runtime; for instance, users of Ray, Spark,
   or PyTorch Distributed must work on the head node for development. Hosted notebook services often serve as stop-gaps for this issue.
   Runhouse allows Python to orchestrate these systems remotely, bringing the development workflow back to standard Python.
#. **Infrastructure Management**: Runhouse captures infrastructure as code, providing a clear contract between the application
   and infrastructure, saving ML teams from having to learn the intricacies of networking, security, and DevOps.

ML Workflow with and without Runhouse
-------------------------------------
.. list-table::
   :widths: 25 35 40
   :header-rows: 1

   * - Aspect
     - Without Runhouse
     - With Runhouse
   * - **Development / Research**
     - Researchers start in hosted notebooks or SSH'ed into a cluster:

       - Fast and interactive development
       - But usually non-standard compute environment and code
     - Researchers write normal code:

       - Each dispatch takes <5 seconds, providing interactive development experience
       - Code executes on the same compute and environment of production
       - Logs stream back to local
   * - **Research to Production**
     - Research to production happens over the course of days or weeks:

       - Notebook code needs translation to orchestrator nodes
       - Most time spent waiting to rebuild and resubmit pipelines
       - Each iteration loop takes about 20+ minutes
     - Moving to production is instant:

       - Orchestrator nodes contain 5 lines of dispatch code
       - Orchestrators are used to schedule, log, and monitor runs
   * - **Debugging and Updating**
     - Production debugging is challenging:

       - Orchestrators designed for scheduling and logging runs
       - Not development-friendly runtimes
       - Continue "debug through deployment"
     - Easily debug or update pipelines in production:

       - Branch the underlying code
       - Make changes and dispatch iteratively
       - Merge back into main



Using Runhouse with Common ML Tools and Libraries
---------------------------------------------
Runhouse is built to be extremely unopinionated and work closely with familiar tools and libraries in the ML ecosystem.
Its APIs are similar to those of other systems, making it easy to compare and contrast. In many cases, Runhouse complements or extends existing tools,
while in other instances, it can replace them entirely.

Notebooks and IDEs (Hosted or Local)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
ML engineers often lose the ability to develop locally when dataset sizes and the need for accelerated compute exceed the capabilities of local hardware.
Hosted notebooks have become a common solution for rapid iteration during the research phase, but they can fragment the development workflow and introduce the
"research-to-production" gap that doesn't exist in traditional software engineering.

Runhouse advocates for defining ML programs and pipelines with standard Python code. Classes and functions should be portable, importable, testable,
and managed using software best practices within a team repository. Code development is best done in a traditional IDE,
but Runhouse is flexible about how the code is executed interactively. You can launch compute, dispatch code, and execute it in script form or use Notebooks as interactive shells.

Since Runhouse code is being dispatched to remote clusters, classes are actually remote objects that can be accessed by mutli-threaded calls. If we instantiate a remote
trainer class and launch training loops in one *local* thread, we can actually make a separate connection to the remote object in another thread. This way, we can
simultaneously do multiple things at the same time with the same remote object: I might want to simultaneously to the training epochs also save model checkpoints down
and run test evaluations. This can be done from three scripts, or three notebooks cells.

We show here how a LoRA Fine Tuner class can be launched from a notebook
in `this example <https://github.com/run-house/runhouse/tree/1b047c9b22839c212a1e2674407959e7e775f21b/examples/lora-example-with-notebook>`_.

Workflow orchestrators (e.g. Airflow, Prefect, Dagster, Flyte, Metaflow, Argo)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Workflow orchestrators are excellent for monitoring, telemetry, fault tolerance, and scheduling, so we recommend using them for these tasks.
However, they shouldn't act as your application or runtime. For example, instead of converting a Python application into an Airflow DAG to run
training on a GPU, you simply make an HTTP call within an Airflow step to trigger a training function running as a service.

By avoiding the need to repack ML code into pipelines, teams can significantly reduce research-to-production time and improve debuggability.
Runhouse ensures that the code committed to the team repository will execute reproducibly in production without additional translation.
Additionally, iteration loops remain fast in production, whether for debugging or further development.
ML engineers can reproduce a failed production run locally by copying the dispatch code, quickly debug and iterate, then push the changes.
This approach is much faster than the traditional 20+ minute cycles required to rebuild and rerun orchestrator pipelines.

There's many clever patterns that Runhouse enables in conjunction with orchestrators that saves time and money.

* Reusing of the same compute across multiple tasks, while separating the steps in the orchestrator for clarity. For instance, avoiding the I/O overhead of repeatedly writing/reading data for each step of an Argo/Kubeflow pipeline.
* Sharing a single service to be shared across multiple orchestrator pipelines. For instance, a single embeddings service can be used by multiple pipelines.
* Maintaining a single orchestrator, but dispatch each pipeline step to aribitrary clusters, regions, or even clouds. For instance, do pre-processing on AWS, but GPU training on GCP where you have quota/credits.
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

Serverless frameworks (e.g. Modal, AWS Lambda)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Serverless frameworks enable on-the-fly service allocation but often require pre-packaging or command-line interface (CLI) launches outside of
standard Python environments. Runhouse, on the other hand, runs entirely within a Python interpreter, allowing it to extend the
compute capabilities of existing Python applications and allocate resources within your own infrastructure.

Many serverless solutions aren't suitable for ML workloads. For instance, AWS Lambda struggles with large datasets, GPU-accelerated tasks,
or long-running jobs. Runhouse can offload these tasks to ephemerally launched, but powerful compute that lasts until the job is done.
Even when evaluating serverless solutions optimized for ML, it's essential to distinguish between those optimized for inference and Runhouse.
For inference, you likely prioritize latency, cold start times and typically execute on a few limited types of hardware.
But if you are considering executing recurring training for instance, Runhouse is significantly more optimized; you have better hardware heterogeneity,
debuggability, statefulness across epochs, and the ability to efficiently use compute.

Slurm-Style Compute Interfaces (e.g. Slurm, SkyPilot, Mosaic, SageMaker Training)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
In this category of Slurm-style solutions, compute is allocated on the fly and scripts are used as entry points.
For heavyweight jobs which are run manully, such as a research lab training a large language
model over hundreds of GPUs, this style of execution works quite well. However, for recurring enterprise ML use cases, there are several distinct disadvantages
that Runhouse fixes.

* Limited control over execution flow, making it difficult to dispatch multiple stages or function calls to the same compute resource (e.g., loading datasets, training, and evaluation).
* Weak fault tolerance due to the inability to catch and handle remote exceptions (all exception handling must occur within the script, leaving little recourse for issues like out-of-memory errors)
* Configuration sprawl as training scripts branch for each new method or experiment, and combinations of settings that work together grow sparser and sparser.

For elastic compute scenarios, Runhouse uses SkyPilot to allocate resources but goes beyond that by offering (re)deployment and execution management.
This restores control over execution, adds fault tolerance, and allows all compute configurations to be defined in code.
