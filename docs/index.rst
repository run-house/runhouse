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

Runhouse enables rapid, cost-effective machine learning development.

With Runhouse, your ML code executes "serverlessly." Dispatch Python functions and classes to any of your own cloud compute infrastructure, and call them
eagerly as if they were local. This means:

#. You can natively run and debug your code on remote GPUs or other powerful infra, like Ray, Spark, or Kubernetes, from your local IDE.
#. This code then runs as-is in CI/CD or production, with no research-to-production delays, where the underlying code will be identically dispatched and executed on ephemeral Runhouse clusters.
#. Hardware requirements and environment is captured in code and you gain fine-grained control to more efficiently bin pack jobs and utilize multiple clusters or even cluods.

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

   why-runhouse
   tutorials/quick-start-cloud
   tutorials/quick-start-den
   how-to-use-runhouse


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
