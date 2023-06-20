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

PyTorch lets you send a model or tensor :code:`.to(device)`, so why can't you do :code:`my_fn.to('a_gcp_a100')`,
or :code:`my_table.to('parquet_in_s3')`?
Runhouse allows just that: send code and data to any of your compute or data infra (with your own cloud creds),
all in Python, and continue to use them eagerly exactly as they were.

Runhouse lets you:

* Natively program across compute resources
* Seamlessly command data between storage and compute
* Access resources across environments and users
* Shared resources among teams as living assets


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
   Pipelining: BERT <https://github.com/run-house/tutorials/tree/stable/t05_BERT_pipeline>

.. toctree::
   :maxdepth: 1
   :caption: Additional Resources

   debugging_logging
   data_collection
   Source Code <https://github.com/run-house/runhouse>
   REST API Guide <https://api.run.house/docs>
   Dashboard <https://www.run.house/dashboard>
   Funhouse <https://github.com/run-house/funhouse>


Contributing and Community
--------------------------
- `Issue Tracker <https://github.com/run-house/runhouse/issues/>`_
- `Contributing <https://github.com/run-house/runhouse/blob/main/CONTRIBUTING.md>`_
