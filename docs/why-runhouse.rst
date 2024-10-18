Why Runhouse
=====================

Runhouse is a Python library that allows any application to flexibly and powerfully utilize remote compute
infrastructure by deploying and calling remote services on the fly. Simply put, Runhouse enables "serverless" execution on your own infrastructure.

Runhouse is principally designed for Machine Learning-style workloads (online, offline, training, and inference), where the need for heterogeneous
remote compute is frequent and flexibility is paramount to minimize costs.

Key Benefits
------------

Runhouse solves a few major problems for machine learning and AI teams:

#. **Iterability and Debuggability**: For developers, Runhouse allows fast local iteration loops with deployment-and-execution loops of <2 seconds. Using Runhouse to develop feels local-like. And because Runhouse enables identical excution in production, this means reproducing production errors and debugging them is similarly easy.
#. **Cost**: Runhouse introduces the flexibility to allocate compute only while needed, right-size instances based on
   the size of the workload, work across multiple regions or clouds for lower costs, and share compute and services
   across tasks. Users typically see cost savings on the order of 50-75%, depending on the workload.
#. **Development at scale**: Powerful, GPU accelerated hardware or distributed clusters (Spark, Ray) can be
   disruptive to adopt. All development, debugging, automation, and deployment to occur on their runtime. For instance, users of Ray, Spark,
   or PyTorch distributed must be tunneled into the head node at all times for development. As a stop-gap, there has been a proliferation of hosted notebook services.
   Runhouse allows Python to orchestrate to these systems remotely, returning the development workflow and operations to standard Python.
#. **Infrastructure overhead**: Runhouse thoughtfully captures infrastructure in code, providing a clear
   contract between the application and infrastructure, and saving ML teams from learning all the networking,
   security, and DevOps underneath.

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
Runhouse's APIs bear similarity to other systems, so it's helpful to compare and contrast. In many cases,
Runhouse is not a replacement for these systems but rather a complement or extension. In others, you may be able
to replace your usage of the other system entirely with Runhouse.

Notebooks and IDEs (Hosted or Local)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
ML engineers lost the ability to do local development as dataset size and requirements for accelerated compute exceeded what was feasible to have locally.
Hosted notebooks were a natural answer to enable the rapid iteration required at the research stage. This has led to a fragementation of the development workflow,
and introduced the "research-to-production" concept that does not exist in traditional software engineering.

Runhouse believes the best pattern is using regular Python code to define your ML programs and pipelines. All of your classes and functions should
be portable, importable, testable, and be managed through software best practices in a team repository. The underlying code is best developed in a traditional IDE.
But if the user wants to interactively *execute* these underlying pieces of code, then Runhouse is signficiantly less opinionated. You can launch compute, dispatch code,
and execute script-style or use notebooks as interactive shells.

Since Runhouse code is being dispatched to remote clusters, classes are actually remote objects that can be accessed by mutli-threaded calls. If we instantiate a remote
class, and for instance, launch training loops in one *local* threads, we can actually make a separate connection to the remote object in another thread. This way, we can
simultaneously do multiple things at the same time with the same remote object -- for instance, I might want to simultaneously to the training epochs also save model checkpoints down
and run test evaluations. This can be done from three scripts, or three notebooks cells.

We show here how a LoRA Fine Tuner class can be launched from a notebook
in `this example <https://github.com/run-house/runhouse/tree/1b047c9b22839c212a1e2674407959e7e775f21b/examples/lora-example-with-notebook>`_.

Workflow orchestrators (e.g. Airflow, Prefect, Dagster, Flyte, Metaflow, Argo)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Generally, workflow orchestrators are good at monitoring, telemetry, fault-tolerance, and scheduling, so
we recommend using an orchestrator for those features. However, orchestrators shouldn't be your application or runtime.
For example, consider the difference between converting your Python application into an Airflow DAG to run the training portion on a GPU,
versus making an HTTP call within the Airflow step to a training function running as a service on a GPU.

By avoiding the need to repack ML code into the pipelines, teams greatly benefit from shortened research-to-production time and improved debuggability.
Runhouse ensures that the research code committed to the team repo will reproducibly execute in production, with no additional translation needed.
Additionally, iteration loops are remain fast in production, whether in debugging or additional development. ML engineers can exactly reproduce a failed
production run locally by copying the dispatch code, debug and iterate on the code rapidly, and then push changes. This is much faster than traditional
iteration loops which take 20+ minutes to rebuild and rerun orchestrator pipelines.

For example, with Runhouse it's easy to allocate small compute to start a training but if the training fails due to OOM
restart it with a slightly larger box. Other compute flexibility like multi-region or multi-cloud which other
orchestrators struggle with are trivial for Runhouse.

There's many clever patterns that Runhouse enables in conjunction with orchestrators that saves time and money.

* Reuse of the same compute across multiple nodes (while separating the steps in orchestrators for clarity). For instance, avoid I/O costs of repeatedly writing/reading data every step.
* Share a single service to be shared across multiple orchestrator pipelines. For instance, a single embeddings service can be used by multiple pipelines.
* Maintain a single orchestrator, but dispatch each pipeline step to aribitrary clusters, regions, or even clouds. For instance, do pre-processing on AWS, but GPU training on GCP where you have quota/credits.
* Catch and handle errors natively from the orchestrator node, since the orchestrator runtime is a Python-based driver for the execution - for instance, on fail due to OOM, launch a larger box and rerun.

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
Serverless frameworks allow for the allocation of services on the fly but within a well-defined sandbox, and not
strictly from within regular Python - they require specific pre-packaging or CLI launch
commands outside Python. Runhouse runs fully in a Python interpreter so it can extend the compute power of practically
any existing Python application, and allocates services inside your own compute, wherever that may be. We may even
support serverless systems as compute backends in the future.

As a practical matter, we find that many serverless solutions are not well suited to all ML workloads. For instance, AWS Lambdas
will struggle with large datasets, GPU accelerated execution, or long-running tasks. Runhouse can offload these tasks to ephemerally launched
but long-lasting elastic/Kubernetes compute until they are completed. For serverless solutions designed for ML, it is important to distinguish
between solutions **optimized for inference** vs. Runhouse. For inference, you care a lot about latency, cold start times and typically execute
on a few specific types of hardware. But take recurring training for instance - Runhouse is significantly more optimized for training where
you care about iterability, debuggability, and efficient utilization of compute

Slurm-Style Compute Interfaces (e.g. Slurm, SkyPilot, Mosaic, SageMaker Training)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
What we characterize as a Slurm-style solution is characterized by using scripts as entry points. These tools can allocate compute on the fly
with various levels of granularity of control over required resources. For jobs which are heavyweight and manual, such as a research lab training a large language
model over hundreds of GPUs, this style of execution works quite well. However, for recurring enterprise ML use cases, there are several distinct disadvantages
that Runhouse attempts to fix.

* Limited control over execution flow, such as dispatching multiple workflow stages or function calls to the same compute resource (e.g., loading the dataset, training for an epoch, and evaluating)
* Weak fault tolerance due to the inability to catch and handle remote exceptions (all exception handling must occur within the script, leaving little recourse for issues like out-of-memory errors)
* Configuration sprawl as training scripts branch for each new method or experiment, and combinations of settings that work together grow sparser and sparser.

For certain use cases like launching elastic compute, Runhouse uses SkyPilot to allocate compute. However, Runhouse goes beyond resource allocation, and
includes (re)deployment and management of execution to give back control over execution, add fault tolerance, and define all compute/config in code.
