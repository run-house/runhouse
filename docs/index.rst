.. runhouse documentation master file, created by

Runhouse 🏃‍♀️🏠
====================================

.. raw:: html

   <p style="text-align:left">
   <strong>Programmable remote compute and data across environments and users</strong>
   </p>

   <p style="text-align:left">
   <a class="reference external image-reference" style="vertical-align:9.5px" href="https://discord.gg/RnhB6589Hs"><img alt="Join Discord" src="https://img.shields.io/discord/1065833240625172600?label=Discord&style=for-the-badge"></a>
   <script async defer src="https://buttons.github.io/buttons.js"></script>
   <a class="github-button" href="https://github.com/run-house/runhouse" data-show-count="true" data-size="large" aria-label="Star run-house/runhouse on GitHub">Star</a>
   <a class="github-button" href="https://github.com/run-house/runhouse/subscription" data-icon="octicon-eye" data-size="large" aria-label="Watch run-house/runhouse on GitHub">Watch</a>
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

.. warning::
    🚨 **This is an Alpha** 🚨

    Runhouse is heavily under development. We are sharing it with a few select people to collect feedback, and expect to iterate on the APIs considerably before reaching beta (version 0.1.0).


Getting Started 🐣
------------------------------------
.. code-block:: console

    $ pip install runhouse
    # Or "runhouse[aws]", "runhouse[gcp]", "runhouse[azure]", "runhouse[all]"
    $ sky check
    # Optionally, for portability (e.g. Colab):
    $ runhouse login

.. tip::
   See the :ref:`Installation` section for more detailed instructions.


Tutorials 👨‍🏫
------------------------------------
Our `tutorials <https://github.com/run-house/tutorials/>`_ have been structured to provide a comprehensive walk through of the APIs,
as well as introduce you to the tools and usage patterns of Runhouse.
We've devised them to chart a fun path through our features.


Contribute 👷‍♀️
------------------------------------
We'd love for you to contribute to Runhouse! Please reach out to us with any questions or support requests 🙂

- `Issue Tracker <https://github.com/run-house/runhouse/issues/>`_
- `Source Code <https://github.com/run-house/runhouse/>`_

Join the Community!
------------------------------------
- `Discord <https://discord.gg/RnhB6589Hs/>`_
- `Twitter <https://twitter.com/runhouse_/>`_


Documentation
--------------------

.. toctree::
   :maxdepth: 1
   :caption: Getting Started

   tutorials/common_use_cases
   tutorials/getting_started


.. toctree::
   :maxdepth: 1
   :caption: Using Runhouse

   installation
   cli/cli
   main

   Tutorials <https://github.com/run-house/tutorials>
   REST API Guide <https://api.run.house/docs>
   Dashboard <https://api.run.house>

.. toctree::
   :maxdepth: 1
   :caption: Security

   secrets/secrets
   access_controls/access_controls
