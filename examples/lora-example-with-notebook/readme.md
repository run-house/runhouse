## LoRA Fine-Tuning Class with Example of Notebook Usage
In this example, we define a Fine Tuner class (LoraFineTuner.py) in **regular Python** and launch remote GPU compute to do the fine-tuning.

In particular, we show how you can start the fine tuning and interact with the fine-tuning class (a remote object) through regular Python or a Notebook. Runhouse lets you work *locally* with *remote objects* defined by regular code and edited locally, compared to tooling like hosted notebooks which let you *work locally while SSH'ed into a remote setting.* This offers a few distinct advantages:
* **Real compute and real data:** ML Engineers and data scientists do not need to launch projects on toy compute offered in a research environment.
* **Real code:** Rather than working on Notebooks (because they have to), your team is writing code and developing locally just like a normal software team. The only difference is dispatching the work for remote computation since the local machine doesn't have the right hardware.
* **Fast research to production:** The work done while writing and testing the class is essentially enough to bring the work to production as well. There is no costly rebuilding of the same code a second time to work in a Pipeline.
