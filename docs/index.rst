.. runhouse documentation master file, created by

🏃‍♀️Runhouse Overview🏠
====================

.. raw:: html

   <p style="text-align:left">
   <strong>Programmable remote compute and data across environments and users</strong>
   </p>

   <p style="text-align:left">
   <a class="reference external image-reference" style="vertical-align:9.5px" href="https://discord.gg/RnhB6589Hs"><img alt="Join Discord" src="https://img.shields.io/discord/1065833240625172600?label=Discord&style=for-the-badge"></a>
   <script async defer src="https://buttons.github.io/buttons.js"></script>
   <a class="github-button" href="https://github.com/run-house/runhouse" data-show-count="true" data-size="large" aria-label="Star run-house/runhouse on GitHub">Star</a>
   <a class="github-button" href="https://github.com/run-house/runhouse/subscription" data-icon="octicon-eye" data-size="large" aria-label="Watch run-house/runhouse on GitHub">Watch</a>
   <a class="github-button" href="https://github.com/run-house/runhouse/fork" data-icon="octicon-repo-forked" data-size="large" aria-label="Fork run-house/runhouse on GitHub">Fork</a>
   </p>

Runhouse is a unified interface into *existing* compute and data systems, built to reclaim
the 50-75% of ML practitioners' time lost to debugging, adapting, or repackaging code
for different environments.

Runhouse lets you:

* Send functions and data to any of your compute or data infra, all in Python, and continue to
  interact with them eagerly (there's no DAG) from your existing code and environment.
* Share live and versioned resources across environments or teams, providing a unified layer for
  accessibility, visibility, and management across all your infra and providers.

It wraps industry-standard tooling like Ray and the Cloud SDKs (boto, gsutil, etc. via `SkyPilot <https://github.com/skypilot-org/skypilot/>`_
to give you production-quality features like queuing, distributed, async, logging, low latency, hardware efficiency, auto-launching, and auto-termination out of the box.

Who is this for?
----------------

* 🦸‍♀️ **OSS maintainers** who want to improve the accessibility, reproducibility, and reach of their code,
  without having to build support or examples for every cloud or compute system (e.g. Kubernetes) one by one.
  See this in action in 🤗 Hugging Face `Transformers <https://github.com/huggingface/transformers/blob/main/examples/README.md#running-the-examples-on-remote-hardware-with-auto-setup>`_,
  `Accelerate <https://github.com/huggingface/accelerate/blob/main/examples/README.md#simple-multi-gpu-hardware-launcher>`_
  and 🦜🔗 `Langchain <https://python.langchain.com/en/latest/modules/models/llms/integrations/runhouse.html>`_.

* 👩‍🔬 **ML Researchers and Data Scientists** who don't want to spend or wait 3-6 months translating and packaging
  their work for production.

* 👩‍🏭 **ML Engineers** who want to be able to update and improve production services, pipelines, and artifacts with a
  Pythonic, debuggable devX.

* 👩‍🔧 **ML Platform teams** who want a versioned, shared, maintainable stack of services and data artifacts that
  research and production pipelines both depend on.


Table of Contents
-----------------
.. toctree::
   :maxdepth: 1
   :caption: Getting Started

   tutorials/quick_start
   architecture
   api_tutorials

.. toctree::
   :maxdepth: 1
   :caption: API Reference

   api/python
   api/cli

.. toctree::
   :maxdepth: 1
   :caption: Usage Examples

   tutorials/examples/inference
   tutorials/examples/training
   tutorials/examples/distributed
   Pipelining: BERT <https://github.com/run-house/funhouse/tree/main/bert_pipeline>

.. toctree::
   :maxdepth: 1
   :caption: Additional Resources

   debugging_logging
   troubleshooting
   data_collection
   Source Code <https://github.com/run-house/runhouse>
   REST API Guide <https://api.run.house/docs>
   Dashboard <https://www.run.house/dashboard>
   Funhouse <https://github.com/run-house/funhouse>


Contributing and Community
--------------------------
- `Issue Tracker <https://github.com/run-house/runhouse/issues/>`_
- `Contributing <https://github.com/run-house/runhouse/blob/main/CONTRIBUTING.md>`_
