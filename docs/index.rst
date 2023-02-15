.. runhouse documentation master file, created by

Runhouse 🏃‍♀️🏠
============

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

Runhouse is for ML Researchers, Engineers, and Data Scientists who are tired of:

- 🚜 Manually shuttling code and data around between their local machine, remote instances, and cloud storage
- 📤📥 Constantly spinning up and down boxes
- 🐜 Debugging over ssh and notebook tunnels
- 🧑‍🔧 Translating their code into a pipeline DSL just to use multiple hardware types
- 🪦 Debugging in an orchestrator
- 👩‍✈️ Missing out on fancy LLM IDE features
- 🕵️ Struggling to find their teammates' code and data artifacts.


There's no magic yaml, DSL, code serialization, or "submitting for execution." We're just spinning up the cluster for you (or using an existing cluster), syncing over your code, starting a gRPC connection, and running your code on the cluster.

*Runhouse does things for you that you'd spend time doing yourself, in as obvious a way as possible.*

.. warning::
    🚨 **This is an Alpha** 🚨

    Runhouse is heavily under development. We are sharing it with a few select people to collect feedback, and expect to iterate on the APIs considerably before reaching beta (version 0.1.0).


Getting Started 🐣
------------------
.. code-block:: console

    $ pip install runhouse
    # Or "runhouse[aws]", "runhouse[gcp]", "runhouse[azure]", "runhouse[all]"
    $ sky check
    # Optionally, for portability (e.g. Colab):
    $ runhouse login

.. tip::
   See the :ref:`Installation` section for more detailed instructions.


Table of Contents
-----------------
.. toctree::
   :maxdepth: 1
   :caption: Using Runhouse

   installation
   cli/cli
   main

   REST API Guide <https://api.run.house/docs>
   Dashboard <https://api.run.house>

.. toctree::
   :maxdepth: 1
   :caption: Security

   secrets/secrets

.. toctree::
   :maxdepth: 1
   :caption: Runhouse Architecture

   overview/overview
   overview/clusters
   overview/sends
   overview/notebooks
   overview/accessibility
   overview/secrets

.. toctree::
   :maxdepth: 1
   :caption: Tutorials

   10 min Quickstart: Stable Diffusion and FLAN T-5 <https://github.com/run-house/tutorials/tree/main/t01_Stable_Diffusion>
   Dreambooth Training and Inference <https://github.com/run-house/tutorials/tree/main/t02_Dreambooth>
   DALL-E to SD img2img in a Notebook <https://github.com/run-house/tutorials/tree/main/t03_DALLE_SD_pipeline>
   BERT Full Pipeline <https://github.com/run-house/tutorials/tree/main/t05_BERT_pipeline>


Contributing and Community
--------------------------
- `Issue Tracker <https://github.com/run-house/runhouse/issues/>`_
- `Contributing <https://github.com/run-house/runhouse/blob/main/CONTRIBUTING.md>`_
- `Discord <https://discord.gg/RnhB6589Hs/>`_
- `Twitter <https://twitter.com/runhouse_/>`_
