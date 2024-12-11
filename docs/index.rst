🏃‍♀️ Runhouse Docs 🏠
======================

.. raw:: html

   <p style="text-align:left">
   <a class="reference external image-reference" style="vertical-align:9.5px" href="https://discord.gg/RnhB6589Hs"><img alt="Join Discord" width="177px" height="28px" src="https://img.shields.io/discord/1065833240625172600?label=Discord&style=for-the-badge"></a>
   <script async defer src="https://buttons.github.io/buttons.js"></script>
   <a class="github-button" href="https://github.com/run-house/runhouse" data-show-count="true" data-size="large" aria-label="Star run-house/runhouse on GitHub">Star</a>
   <a class="github-button" href="https://github.com/run-house/runhouse/subscription" data-icon="octicon-eye" data-size="large" aria-label="Watch run-house/runhouse on GitHub">Watch</a>
   <a class="github-button" href="https://github.com/run-house/runhouse/fork" data-icon="octicon-repo-forked" data-size="large" aria-label="Fork run-house/runhouse on GitHub">Fork</a>
   </p>

Runhouse enables rapid, cost-effective machine learning development. With Runhouse, your ML code executes "serverlessly."
Dispatch Python functions and classes to any of your own cloud compute infrastructure, and call them
eagerly as if they were local.

Runhouse is specifically designed for machine learning workloads — including online and offline tasks, training, and inference —
where the need for heterogeneous remote compute resources is common, and flexibility is essential to keep development cycles fast and costs low.

Key Benefits
------------

Runhouse solves a few major problems for machine learning and AI teams:

#. **Iterability**: Developing with Runhouse feels like working locally, even if the code is executing on powerful, multi-node remote hardware.
   In research, avoid writing non-standard code in hosted notebooks; in production, don't iterate by building and resubmitting pipelines.
   The team writes standard Python code locally, and it takes less than 2 seconds per iteration to redeploy the code to remote compute.
   The remote filesystem and any unaffected remote objects or functions remain accessible across iterations.
#. **Debuggability**: With Runhouse, there is perfect reproducibility between local and scheduled production execution.
   Research code that works is already production-ready, while any production runs that fail can be debugged locally.
   The combination of identical execution and fast iteration enables a straightforward, rapid debugging loop.
#. **Cost**: Organizations adopting Runhouse utilize their available compute more efficiently, leading to typical cost savings of 50%.
   With Runhouse, ephemeral clusters are allocated only when needed and can be launched across multiple regions or clouds based on quota or cost considerations.
   It's easy to right-size instances based on workload, incorporate spot instances, and even share compute or services on compute across tasks.
#. **Development at Scale**: Adopting powerful, GPU-accelerated hardware or distributed clusters (Spark, Ray) can be
   disruptive. All development, debugging, automation, and deployment to occur on their runtime; for instance, users of Ray, Spark,
   or PyTorch Distributed must work on the head node for development. Hosted notebook services often serve as stop-gaps for this issue.
   Runhouse allows Python to orchestrate these systems remotely, bringing the development workflow back to standard Python.
#. **Infrastructure Management**: Runhouse captures infrastructure as code, providing a clear contract between the application
   and infrastructure, saving ML teams from having to learn the intricacies of networking, security, and DevOps.

ML Workflow with and without Runhouse
-------------------------------------
A quick high-level summary of the differences between developing and deploying ML code with and without Runhouse:

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

Get In Touch
------------
You can join the Runhouse discord, or shoot us a quick note at `hello@run.house <mailto:hello@run.house>`_

Examples
---------
- `Quick Start <https://www.run.house/docs/tutorials/quick-start-cloud>`_
- `PyTorch Training <https://www.run.house/examples/torch-vision-mnist-basic-model-train-test>`_
- `PyTorch Distributed Training <https://www.run.house/examples/pytorch-multi-node-distributed-training>`_
- `Llama3 Fine Tuning <https://www.run.house/examples/fine-tune-llama-3-with-lora>`_
- `Llama3 vLLM Inference <https://www.run.house/examples/run-llama-3-8b-with-vllm-on-gcp>`_

Table of Contents
-----------------
.. toctree::
   :maxdepth: 1
   :caption: Getting Started

   tutorials/quick-start-cloud
   tutorials/quick-start-den
   how-to-use-runhouse
   runhouse-in-your-stack


.. toctree::
   :maxdepth: 1
   :caption: API Basics

   tutorials/api-clusters
   tutorials/api-modules
   tutorials/api-folders
   tutorials/api-secrets
   tutorials/api-resources

.. toctree::
   :maxdepth: 1
   :caption: API Reference

   api/python
   api/cli

.. toctree::
   :maxdepth: 1
   :caption: Other Topics

   tutorials/async.rst
   installation
   debugging-logging
   docker-setup
   docker-workflows
   troubleshooting
   security-and-authentication


Contributing and Community
--------------------------
- `Issue Tracker <https://github.com/run-house/runhouse/issues/>`_
- `Contributing <https://github.com/run-house/runhouse/blob/main/CONTRIBUTING.md>`_
