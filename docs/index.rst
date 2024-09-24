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
   <p>Why did we build Runhouse? To enable fast, debuggable, and iterable development of machine learning workflows.</p>

   <strong>Without Runhouse, the ML workflow is fragmented and slow</strong>

   <ul>
     <li>Researchers start in hosted notebooks or SSH'ed into a cluster:
       <ul>
         <li>Fast and interactive development</li>
         <li>But typically compute, environment, and code all look very different from production</li>
       </ul>
     </li>
     <li>Research to production happens over the course of days or weeks:
       <ul>
         <li>Notebook code needs translation to orchestrator nodes</li>
         <li>Iteration loops take about 20+ minutes, as engineers wait for pipelines to rebuild and rerun</li>
       </ul>
     </li>
     <li>Production debugging is challenging:
       <ul>
         <li>Orchestrators designed for scheduling and logging runs, but are not development-friendly runtimes</li>
         <li>Continue the slow "debug through deployment" process which plagued research to production as well</li>
       </ul>
     </li>
   </ul>

   <strong>With Runhouse, serverless execution of regular code on your own cloud compute enables easy development and deployment</strong>
   <ul>
     <li>Researchers write normal code:
       <ul>
         <li>Each dispatch takes &lt;5 seconds, providing interactive development experience</li>
         <li>Code executes on the same compute and environment as production</li>
         <li>Logs stream back to local</li>
       </ul>
     </li>
     <li>Moving to production is instant:
       <ul>
         <li>Orchestrator nodes contain 5 lines of dispatch code (not 200 lines of application code)</li>
         <li>Rather than being your ML runtime, orchestrators are simply used to schedule, log, and monitor runs</li>
       </ul>
     </li>
     <li>Easily debug or update pipelines in production:
       <ul>
         <li>Branch the underlying code</li>
         <li>Make changes and dispatch iteratively, like in the research step</li>
         <li>Merge back into main when the pipeline runs locally</li>
       </ul>
     </li>
   </ul>

   <p>In short, Runhouse makes ML development feel like normal software development.</p>

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
