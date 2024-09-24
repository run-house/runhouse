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

Why did we build Runhouse? To enable fast, debuggable, and iterable development of machine learning workflows.

**Without Runhouse:** ML workflow is fragmented and development is slow

* Researchers start in hosted notebooks or SSH'ed into a cluster:

  * Fast and interactive development
  * But usually non-standard compute environment and code

* Research to production happens over the course of days or weeks:

  * Notebook code needs translation to orchestrator nodes
  * Most time spent waiting to rebuild and resubmit pipelines, with each iteration loop taking about 20+ minutes

* Production debugging is challenging:

  * Orchestrators designed for scheduling and logging runs, but are not development-friendly runtimes
  * Continue "debug through deployment" that slowed down research to production in the first place

**With Runhouse:** Regular code is dispatched to cloud compute for execution at every step

* Researchers write normal code:

  * Each dispatch takes <5 seconds, providing interactive development experience
  * Code executes on the same compute and environment of production
  * Logs stream back to local

* Moving to production is instant:

  * Orchestrator nodes contain 5 lines of dispatch code (not 500 lines of application code)
  * Rather than being your ML runtime, orchestrators are simply used to schedule, log, and monitor runs

* Easily debug or update pipelines in production:

  * Branch the underlying code
  * Make changes and dispatch iteratively, like in the research step
  * Merge back into main

In short, Runhouse makes ML development feel like normal software development.

Table of Contents
-----------------
.. toctree::
   :maxdepth: 1
   :caption: Getting Started

   tutorials/quick-start-cloud
   tutorials/quick-start-local
   tutorials/quick-start-den
   architecture

.. toctree::
   :maxdepth: 1
   :caption: API Basics

   tutorials/api-clusters
   tutorials/api-modules
   tutorials/api-envs
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
