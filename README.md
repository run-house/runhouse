<h1 align="center">🏃‍♀️ Runhouse 🏠</h1>

[//]: # (<p align="center">)

[//]: # (  <a href="https://discord.gg/RnhB6589Hs"> )

[//]: # (    <img alt="Join Discord" src="https://img.shields.io/discord/1065833240625172600?label=Discord&style=for-the-badge">)

[//]: # (  </a>)

[//]: # (</p>)

## 👵 Welcome Home!

Runhouse is a unified interface into *existing* compute and data systems, built to reclaim
the 50-75% of ML practitioners' time lost to debugging, adapting, or repackaging code
for different environments.

### 🤨 Who is this for?

* 🦸‍♀️ **OSS maintainers** who want to improve the accessibility, reproducibility, and reach of their code,
without having to build support or examples for every cloud or compute system (e.g. Kubernetes) one by one.
   * See this in action in 🤗 Hugging Face ([Transformers](https://github.com/huggingface/transformers/blob/main/examples/README.md#running-the-examples-on-remote-hardware-with-auto-setup), [Accelerate](https://github.com/huggingface/accelerate/blob/main/examples/README.md#simple-multi-gpu-hardware-launcher)) and 🦜🔗 [Langchain](https://python.langchain.com/en/latest/modules/models/llms/integrations/runhouse.html)
* 👩‍🔬 **ML Researchers and Data Scientists** who don't want to spend or wait 3-6 months translating and packaging
their work for production.
* 👩‍🏭 **ML Engineers** who want to be able to update and improve production services, pipelines, and artifacts with a
Pythonic, debuggable devX.
* 👩‍🔧 **ML Platform teams** who want a versioned, shared, maintainable stack of services and data artifacts that
research and production pipelines both depend on.

### 🦾 How does it work?

**_"Learn once, run anywhere"_**

Runhouse is like PyTorch + Terraform + Google Drive.

1. Just as **PyTorch** lets you send a model or tensor `.to(device)`, Runhouse OSS
lets you do `my_fn.to('gcp_a100')` or `my_table.to('s3')`: send functions and data to any of your compute or
data infra, all in Python, and continue to interact with them eagerly (there's no DAG) from your existing code and
environment. Think of it as an expansion pack to Python that lets it take detours to remote
machines or manipulate remote data.
2. Just as **Terraform** is a unified language for creation and destruction of infra, the
Runhouse APIs are a unified interface into existing compute and data systems.
See what we already support and what's on the [roadmap, below](#-roadmap).
3. Runhouse resources can be shared across environments or teams, providing a **Google Drive**-like
layer for accessibility, visibility, and management across all your infra and providers.

This allows you to:
* Call your preprocessing, training, and inference each on different hardware from
inside a single notebook or script
* Slot that script into a single orchestrator node rather than translate it into an ML pipeline DAG of docker images
* Share any of those services or data artifacts with your team instantly, and update them over time

[//]: # (![img.png]&#40;docs/assets/img.png&#41;)
[//]: # (![img_1.png]&#40;docs/assets/img_1.png&#41;)
![img.png](https://raw.githubusercontent.com/run-house/runhouse/main/docs/assets/img.png)
![img_1.png](https://raw.githubusercontent.com/run-house/runhouse/main/docs/assets/img_1.png)

It wraps industry-standard tooling like Ray and the Cloud SDKs (boto, gsutil, etc. via [SkyPilot](https://github.com/skypilot-org/skypilot/))
to give you production-quality features like queuing, distributed, async, logging,
low latency, auto-launching, and auto-termination out of the box.

## 👩‍💻 Enough chitchat, just show me the code

Here is **all the code you need** to stand up a stable diffusion inference service on
a fresh cloud GPU.


```python
import runhouse as rh
from diffusers import StableDiffusionPipeline

def sd_generate(prompt):
    model = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-base").to("cuda")
    return model(prompt).images[0]

gpu = rh.cluster(name="my-a100", instance_type="A100:1").up_if_not()
env = rh.env(reqs=["torch", "diffusers"])
fn_gpu = rh.function(sd_generate).to(system=gpu, env=env)

# the following runs on our remote A100 gpu
sd_generate("An oil painting of Keanu Reeves eating a sandwich.").show()
sd_generate.save(name="sd_generate")
```

On the data side, sync folders, tables, or blobs between local, clusters, and file storage. All
this is done without bouncing off the laptop.

```python
import runhouse as rh

folder_on_gpu = rh.folder(path="./instance_images").to(system=gpu, path="dreambooth/instance_images")

folder_on_s3 = rh.folder(system=gpu, path="dreambooth/instance_images").to("s3", path="dreambooth/instance_images")
folder_on_s3.save()
```

Reuse your saved compute and data resources from anywhere, with a single line of Python.

```python
sd_generate = rh.Function.from_name("sd_generate")
image = sd_generate("A hot dog made of matcha.")

folder_on_s3 = rh.Folder.from_name("dreambooth_outputs")
folder_on_local = folder_on_s3.to("here")
```

These APIs work from anywhere with a Python interpreter and an internet connection.
Notebooks, scripts, pipeline nodes, etc. are all fair game.

## 🚘 Roadmap

Runhouse is an ambitious project to provide a unified API into many paradigms and providers for
various types of infra. You can find our currently support systems and high-level roadmap below.
Please reach out to contribute or share feedback!
- Compute
  - On-prem
    - Single instance - **Supported**
    - Ray cluster - **Supported**
    - Kubernetes - Planned
    - Slurm - Exploratory
  - Cloud VMs
    - AWS - **Supported**
    - GCP - **Supported**
    - Azure - **Supported**
    - Lambda - **Supported**
  - Serverless - Planned
- Data
  - Blob storage
    - AWS - **Supported**
    - GCP - **Supported**
    - R2 - Planned
    - Azure - Exploratory
  - Tables
    - Arrow-based (Pandas, Hugging Face, PyArrow, Ray.data, Dask, CuDF) - **Supported**
    - SQL-style - Planned
    - Lakehouse - Exploratory
  - KVStores - Exploratory
- Management
  - Secrets
    - Runhouse Den (via Hashicorp Vault) - **Supported**
    - Custom (Vault, AWS, GCP, Azure) - Planned
  - RBAC - Planned
  - Monitoring - Planned

## 🙅‍♀️ Runhouse is not

* An orchestrator / scheduler (Airflow, Prefect, Metaflow)
* A distributed compute DSL (Ray, Spark)
* An ML Platform (Sagemaker, Vertex, Azure ML)
* A model registry / experiment manager / MLOps framework (MLFlow, WNB, Kubeflow)
* Hosted compute (Modal, Banana, Replicate)

## 🚨 This is an Alpha 🚨

Runhouse is heavily under development and we expect to iterate
on the APIs before reaching beta (version 0.1.0).

## 🐣 Getting Started

tldr;
```commandline
pip install runhouse
# Or "runhouse[aws]", "runhouse[gcp]", "runhouse[azure]", "runhouse[all]"
sky check
# Optionally, for portability (e.g. Colab):
runhouse login
```

### 🔌 Installation

Runhouse can be installed from Pypi with:
```
pip install runhouse
```

Depending on which cloud providers you plan to use, you can also install the following
additional dependencies (to install the right versions of tools like boto, gsutil, etc.):
```commandline
pip install "runhouse[aws]"
pip install "runhouse[gcp]"
pip install "runhouse[azure]"
# Or
pip install "runhouse[all]"
```

As this is an alpha, we push feature updates every few weeks as new microversions.

### ✈️ Verifying your Cloud Setup with SkyPilot

Runhouse supports both BYO cluster, where you interact with existing compute via their
IP address and SSH key, and autoscaled clusters, where we spin up and down cloud instances
in your own cloud account for you. If you only plan to use BYO clusters, you can
disregard the following.

Runhouse uses [SkyPilot](https://skypilot.readthedocs.io/en/latest/) for
much of the heavy lifting with launching and terminating cloud instances.
We love it and you should [throw them a Github star ⭐️](https://github.com/skypilot-org/skypilot/).

To verify that your cloud credentials are set up correctly for autoscaling, run
```
sky check
```
in your command line. This will confirm which cloud providers are ready to
use, and will give detailed instructions if any setup is incomplete. SkyPilot also
provides an excellent suite of CLI commands for basic instance management operations.
There are a few that you'll be reaching for frequently when using Runhouse with autoscaling
that you should familiarize yourself with,
[here](https://runhouse-docs.readthedocs-hosted.com/en/latest/overview/compute.html#on-demand-clusters).

### 🔒 Creating a Runhouse Account for Secrets and Portability

Using Runhouse with only the OSS Python package is perfectly fine. However,
you can unlock some unique portability features by creating an (always free)
account on [api.run.house](https://api.run.house) and saving your secrets and/or
resource metadata there. For example, you can open a Google Colab, call `runhouse login`,
and all of your secrets or resources will be ready to use there with no additional setup.
Think of the OSS-package-only experience as akin to Microsoft Office,
while creating an account will make your cloud resources sharable and accessible
from anywhere like Google Docs. You
can see examples of this portability in the
[Runhouse Tutorials](https://github.com/run-house/tutorials).

To create an account, visit [api.run.house](https://api.run.house),
or simply call `runhouse login` from the command line (or
`rh.login()` from Python).

> **Note**:
These portability features only ever store light metadata about your resources
(e.g. my_folder_name -> [provider, bucket, path]) on our API servers. All the actual data and compute
stays inside your own cloud account and never hits our servers. The Secrets service stores
your secrets in Hashicorp Vault (an industry standard for secrets management), and our secrets
APIs simply call Vault's APIs. We never store secrets on our API servers. We plan to add
support for BYO secrets management shortly. Let us know if you need it and which system you use.

## 👨‍🏫 Tutorials / API Walkthrough / Docs

[Tutorials can be found here](https://github.com/run-house/tutorials). They have been structured to provide a
comprehensive walkthrough of the APIs.

[Docs can be found here](https://runhouse-docs.readthedocs-hosted.com/en/latest/index.html).
They include both high-level overviews of the architecture and detailed API references.

## 🎪 Funhouse

Check out [Funhouse](https://github.com/run-house/funhouse) for running fun applications using Runhouse --
think the latest Stable Diffusion models, text generation models, launching Gradio spaces, and even more!

## 🙋‍♂️ Getting Help

Please join our [discord server here](https://discord.gg/RnhB6589Hs)
to message us, or email us (first name at run.house), or create an issue.

## 👷‍♀️ Contributing

We welcome contributions! Please check out [contributing](CONTRIBUTING.md) if you're interested.
