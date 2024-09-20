.. runhouse documentation master file, created by

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

Why Runhouse?
-------------
Runhouse was built to accelerate the ML flywheel, by enabling fast, debuggable, and iterable development of pipelines.

Without Runhouse, the workflow is fragmented and iteration is slow. Researchers start in hosted notebooks where there is fast and interactive development, but the enviornment and code are both non-standard. Then that notebook code needs to be translated and packed into orchestrator nodes over days or weeks. This is slow and challenging because each iteration loop takes an half hour, with most time spent waiting to rebuild and resubmit pipelines. In production, debugging errors is also challenging for the same reason: orchestrators were designed to schedule and log runs, not be a development-friendly runtime.

With Runhouse, you are simply dispatching regular code to cloud compute for execution at every step. Researchers are writing normal code and executing it on the same (ephemeral) compute that the code would be run on in production. Each dispatch takes <5 seconds and logs are streaming back to local, giving an interactive development experience. Moving to production means moving the 5 lines of dispatch code (not 500 lines of applciation code) into orchestrator nodes. And in production, debugging or further improving pipelines is just branching the underlying application code, executing the same dispatch code, and merging back into main. In short, Runhouse makes ML development feel like normal software development.

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
