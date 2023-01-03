.. runhouse documentation master file, created by
   sphinx-quickstart on Thu Nov 24 01:40:52 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Runhouse 🏃‍♀️🏠
====================================

.. raw:: html

   <p style="text-align:left">
   <strong>A multiplayer cloud compute and data environment, like Google Docs for ML</strong>
   </p>

   <p style="text-align:left">
   <a class="reference external image-reference" style="vertical-align:9.5px" href="https://join.slack.com/t/runhouse/shared_invite/zt-1j7pwsok1-vQy0Gesh55A2fPyyEVq8nQ"><img src="https://img.shields.io/badge/Runhouse-Join%20Slack-fedcba?logo=slack" style="height:27px"></a>
   <script async defer src="https://buttons.github.io/buttons.js"></script>
   <a class="github-button" href="https://github.com/run-house/runhouse" data-show-count="true" data-size="large" aria-label="Star run-house/runhouse on GitHub">Star</a>
   <a class="github-button" href="https://github.com/run-house/runhouse/subscription" data-icon="octicon-eye" data-size="large" aria-label="Watch run-house/runhouse on GitHub">Watch</a>
   </p>

Runhouse breaks down the silos between machine learning compute and data environments, and makes interacting with cloud resources easy and Pythonic:

With Runhouse, you can:

- Run effortlessly on any hardware in experimentation, research, or production, iterating and debugging like you're running locally on bare metal.
- Scale up, productionize, or share your work without weeks of packaging or translating into platform DSLs.
- Retain control of their underlying infra, customizing it to your unique needs and continuing to use any ML tools you would from cloud instances.


.. warning::
   🚨 **Caution: This is an Unstable Semi-Private Preview** 🚨

   Runhouse is heavily under development and unstable. We are quite a ways away from having our first stable release. We are sharing it privately with a few select people to collect feedback, and expect a lot of things to break off the bat.

   If you would be so kind, we would love if you could have a notes doc open as you install and try Runhouse for the first time. Your first impressions, pain points, and highlights are very valuable to us.

Contribute
------------------------------------
We'd love for you to contribute to Runhouse! Please reach out to us with any questions or support requests 🙂

- `Issue Tracker <https://github.com/run-house/runhouse/issues/>`_
- `Source Code <https://github.com/run-house/runhouse/>`_

Join the Community!
------------------------------------
- Slack: `Coming Soon`...
- Discord: `Coming Soon`...
- Twitter: `Coming Soon`...



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

   cli/cli
   main

   REST API Guide <https://api.run.house/docs>
   Dashboard <https://api.run.house>
   faqs
   roadmap/roadmap


.. toctree::
   :maxdepth: 1
   :caption: Security

   secrets/secrets
   access_controls/access_controls
   data_policy/data_policy