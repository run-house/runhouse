.. runhouse documentation master file, created by

🏃‍♀️Runhouse Overview🏠
====================

.. raw:: html

   <p style="text-align:left">
   <strong>Programmable remote compute and data across environments and users</strong>
   </p>

   <p style="text-align:left">
   <a class="reference external image-reference" style="vertical-align:9.5px" href="https://discord.gg/RnhB6589Hs"><img alt="Join Discord" width="177px" height="28px" src="https://img.shields.io/discord/1065833240625172600?label=Discord&style=for-the-badge"></a>
   <script async defer src="https://buttons.github.io/buttons.js"></script>
   <a class="github-button" href="https://github.com/run-house/runhouse" data-show-count="true" data-size="large" aria-label="Star run-house/runhouse on GitHub">Star</a>
   <a class="github-button" href="https://github.com/run-house/runhouse/subscription" data-icon="octicon-eye" data-size="large" aria-label="Watch run-house/runhouse on GitHub">Watch</a>
   <a class="github-button" href="https://github.com/run-house/runhouse/fork" data-icon="octicon-repo-forked" data-size="large" aria-label="Fork run-house/runhouse on GitHub">Fork</a>
   </p>

Runhouse is a Python framework for composing and sharing production-quality backend apps and services ridiculously quickly and on your own infra.

Runhouse is built to do four things:

1. Make it easy to send an arbitrary block of your code - function, subroutine, class, generator - to run on souped up
   remote infra. It's basically a flag flip.
2. Eliminate CLI and Flask/FastAPI boilerplate by allowing you to send your function or class directly to your remote
   infra to execute or serve, and keep them debuggable like the original code, not a ``subprocess.Popen`` or
   Postman/ ``curl`` call.
3. Bake-in the middleware and automation to make your app production-quality, secure, and sharable instantly. That means
   giving you out of the box, state-of-the-art auth, HTTPS, telemetry, packaging, developer tools and deployment
   automation, with ample flexibility to swap in your own.
4. Bring the power of `Ray <https://www.ray.io/>`_ to any app, anywhere, without having to learn Ray or manage Ray clusters,
   like `Next.js <https://nextjs.org/>`_ did for React. OpenAI, Uber, Shopify, and many others use Ray to power their ML
   infra, and Runhouse makes its best-in-class features accessible to any project, team, or company.

Who is this for?
----------------

* 👩‍🔧 **Engineers, Researchers and Data Scientists** who don't want to spend 3-6 months translating and packaging their work to share it, and want to be able to iterate and improve production services, pipelines, experiments and data artifacts with a Pythonic, debuggable devX.

* 👩‍🔬 **ML and data teams** who want a versioned, shared, maintainable stack of services used across research and production, spanning any cloud or infra type (e.g. Instances, Kubernetes, Serverless, etc.).

* 🦸‍♀️ **OSS maintainers** who want to supercharge their setup flow by providing a single script to stand up their app on any infra, rather than build support or guides for each cloud or compute system (e.g. Kubernetes) one by one.
  See this in action in 🤗 Hugging Face `Transformers <https://github.com/huggingface/transformers/blob/main/examples/README.md#running-the-examples-on-remote-hardware-with-auto-setup>`_,
  `Accelerate <https://github.com/huggingface/accelerate/blob/main/examples/README.md#simple-multi-gpu-hardware-launcher>`_
  and 🦜🔗 `Langchain <https://python.langchain.com/en/latest/modules/models/llms/integrations/runhouse.html>`_.


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
   security_and_authentication
   Source Code <https://github.com/run-house/runhouse>
   REST API Guide <https://api.run.house/docs>
   Runhouse Den Dashboard <https://www.run.house/dashboard>
   Funhouse <https://github.com/run-house/funhouse>


Contributing and Community
--------------------------
- `Issue Tracker <https://github.com/run-house/runhouse/issues/>`_
- `Contributing <https://github.com/run-house/runhouse/blob/main/CONTRIBUTING.md>`_
