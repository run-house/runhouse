Why Runhouse
=====================

Runhouse is a Python library that allows any application to flexibly and powerfully utilize remote compute
infrastructure by deploying and calling remote services on the fly. Simply put, Runhouse enables "serverless" execution on your own infrastructure.

Runhouse is principally designed for Machine Learning-style workloads (online, offline, training, and inference), where the need for heterogeneous
remote compute is frequent and flexibility is paramount to minimize costs.


.. list-table:: Comparison of ML Workflow
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

Key Benefits
------------

Runhouse solves a few major problems for machine learning and AI teams:

#. **Iterability and Debuggability**:
#. **Cost**: Runhouse introduces the flexibility to allocate compute only while needed, right-size instances based on
   the size of the workload, work across multiple regions or clouds for lower costs, and share compute and services
   across tasks. Users typically see cost savings on the order of 50-75%, depending on the workload.
#. **Development at scale**: Powerful, GPU accelerated hardware or distributed clusters (Spark, Ray) can be
   disruptive to adopt. All development, debugging, automation, and deployment to occur on their runtime. For instance, users of Ray, Spark,
   or PyTorch distributed must be tunneled into the head node at all times for development. As a stop-gap, there has been a proliferation of hosted notebook services.
   Runhouse allows Python to orchestrate to these
   systems remotely, returning the development workflow and operations to standard Python. Teams using Runhouse
   can even abandon hosted development notebooks and sandboxes entirely, again saving considerable cost and
   research-to-production time.
#. **Infrastructure overhead**: Runhouse thoughtfully captures infrastructure in code, providing a clear
   contract between the application and infrastructure, and saving ML teams from learning all the networking,
   security, and DevOps underneath.

Runhouse + ML Tools and Libraries
---------------------------------


Runouse  calling the heterogeneous portions as remote services.
By contrast, incorporating heterogeneous compute into the runtime of an application, as workflow orchestrators (e.g., Airflow, Prefect)
or distributed libraries (e.g., Ray, Spark) do, is extremely disruptive and inflexible at every level— development workflow,
debugging, DevOps, and infrastructure.
For example, consider the difference between converting your Python application into an Airflow DAG to run the training portion on a GPU,
versus making an HTTP call within the application to a training function running as a service on a GPU.

While calling a function or class as a remote service is a common pattern (i.e., microservices),
it divides the code into multiple applications, multiplying the DevOps overhead, as each requires its own configuration,
automation, scaling, etc. Runhouse combines the best of both approaches: providing limitless compute dynamism and
flexibility in Python without disrupting the runtime or fragmenting the application, by offloading functions and classes to remote compute as services on the fly.


Runhouse's APIs bear similarity to other systems, so it's helpful to compare and contrast. In many cases,
Runhouse is not a replacement for these systems but rather a complement or extension. In others, you may be able
to replace your usage of the other system entirely with Runhouse.

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


Workflow orchestrators (e.g. Airflow, Prefect, Dagster, Flyte, Metaflow, Argo)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Generally, workflow orchestrators are good at monitoring, telemetry, fault-tolerance, and scheduling, so
we recommend using an orchestrator for those features. However, orchestrators shouldn't be your application or runtime.
Runhouse is used within each pipeline node to specify the required compute, dispatch underlying program code, and execute the code remotely.

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

Serverless frameworks (e.g. Modal, AWS Lambda)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Serverless frameworks allow for the allocation of services on the fly but within a well-defined sandbox, and not
strictly from within regular Python - they require specific pre-packaging or CLI launch
commands outside Python. Runhouse runs fully in a Python interpreter so it can extend the compute power of practically
any existing Python application, and allocates services inside your own compute, wherever that may be. We may even
support serverless systems as compute backends in the future.

Infrastructure in code (e.g. SkyPilot, Pulumi)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Infrastructure in code tools allocate compute on the fly but can't utilize it instantly
to offload execution within the application (though you could call a predefined script entrypoint or API
endpoint). Runhouse uses SkyPilot to allocate compute but is vertically integrated to be able
to perform allocation, (re)deployment, and management of a new service all in Python so the new compute can be used
instantly within the existing application. It also doesn't need to perform allocation to create new services -
it can use existing compute or static VMs.

GPU/Accelerator dispatch (e.g. PyTorch, Jax, Mojo)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
GPU/Accelerator dispatch systems give the ability to offload computation to a local GPU or
TPU. Runhouse does not have this capability but can offload a function or class to a remote instance with an
accelerator, which can then itself use libraries like PyTorch or Jax (and maybe one day Mojo) to use the accelerator.
