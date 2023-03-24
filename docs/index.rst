.. runhouse documentation master file, created by

Runhouse 🏃‍♀️🏠
===================

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

Runhouse has four top-level objectives:


* Allowing users to natively program across compute resources
* Allowing users to command data between storage and compute
* Making resources accessible across environments and users
* Allowing resources to be shared among teams as living assets

Keep reading on to see how Runhouse achieves this, or explore our
:ref:`Architecture Section <High-level Architecture>`, :ref:`Python API`, and `Tutorials <https://github.com/run-house/tutorials>`_.

.. warning::
    **This is an Alpha:**
    Runhouse is heavily under development and we expect to iterate on the APIs before reaching beta (version 0.1.0).


Getting Started 🐣
------------------
.. code-block:: console

    $ pip install runhouse
    # Or "runhouse[aws]", "runhouse[gcp]", "runhouse[azure]", "runhouse[all]"

Please check out the :ref:`Installation` section for more detailed instructions.


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
   Funhouse <https://github.com/run-house/funhouse>

.. toctree::
   :maxdepth: 1
   :caption: Security

   secrets/secrets
   secrets/vault_secrets
   data_collection


.. toctree::
   :maxdepth: 1
   :caption: Runhouse Architecture

   overview/high_level
   overview/compute
   overview/data
   overview/accessibility
   overview/management

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
