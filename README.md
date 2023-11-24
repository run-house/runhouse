# 🏃‍♀️ Runhouse 🏠

[![Discord](https://dcbadge.vercel.app/api/server/RnhB6589Hs?compact=true&style=flat)](https://discord.gg/RnhB6589Hs)
[![Twitter](https://img.shields.io/twitter/url/https/twitter.com/runhouse_.svg?style=social&label=@runhouse_)](https://twitter.com/runhouse_)
[![Website](https://img.shields.io/badge/run.house-green)](https://www.run.house)
[![Docs](https://img.shields.io/badge/docs-blue)](https://www.run.house/docs)
[![Den](https://img.shields.io/badge/runhouse_den-purple)](https://www.run.house/login)


## 👵 Welcome Home!

Runhouse is a Python framework for composing and sharing production-quality backend apps and services _ridiculously
quickly_ on any infra. Running the following will stand up a microservice on a fresh AWS EC2 box:

```python
import runhouse as rh

def run_home(name: str):
    return f"Run home {name}!"

if __name__ == "__main__":
    cpu_box = rh.ondemand_cluster(name="my-cpu", instance_type="CPU:2", provider="aws")
    remote_fn = rh.function(run_home).to(cpu_box)
    print(remote_fn("Jack"))
```

This is regular Python running on your own machine(s) with your existing code. There's no YAML, no magic CLI
incantations, no decorators, and no prior setup (other than `pip install runhouse` and having `~/.aws/credentials` in
this case).

Runhouse is built to do four things:
1. Bring the power of Ray to any app, anywhere, without having to learn Ray or manage Ray clusters, like Next.js did
for React. OpenAI, Uber, Shopify, and many others use Ray to power their ML infra, and Runhouse makes its best-in-class
features accessible to any project, team, or company.
2. Make it easy to send an arbitrary block of your code - function, subroutine, class, generator, whatever -
to run on souped up remote infra. It's basically a flag flip.
3. Eliminate CLI and Flask/FastAPI boilerplate by allowing you to send your function or class directly to remote
infra to execute or serve, and keep them debuggable like the original code, not a subprocess.Popen or postman/curl call.
4. Bake-in the middleware and automation to make your app production-quality, secure, and sharable instantly.
That means giving you best-of-breed auth, HTTPS, telemetry, packaging, and deployment automation, with ample
flexibility to swap in your own.

## 🤨 Who is this for?

* 👩‍🔧 **Engineers, Researchers and Data Scientists** who don't want to spend 3-6 months translating and packaging
their work to share it, and want to be able to iterate and improve production services, pipelines, and artifacts
with a Pythonic, debuggable devX.
* 👩‍🔬 **ML and data teams** who want a versioned, shared, maintainable stack of services used across
research and production.
* 🦸‍♀️ **OSS maintainers** who want to supercharge their setup flow by providing a single script to stand up their app
on any infra, rather than build support or guides for each cloud or compute system (e.g. Kubernetes) one by one.
   * See this in action in 🤗 Hugging Face ([Transformers](https://github.com/huggingface/transformers/blob/main/examples/README.md#running-the-examples-on-remote-hardware-with-auto-setup), [Accelerate](https://github.com/huggingface/accelerate/blob/main/examples/README.md#simple-multi-gpu-hardware-launcher)) and 🦜🔗 [Langchain](https://python.langchain.com/en/latest/modules/models/llms/integrations/runhouse.html)

## 🦾 How does it work?

Suppose you create a cluster object:

```python
import runhouse as rh

gpu = rh.cluster(name="my-a100", host=my_cluster_ip, ssh_creds={"user": "ubuntu", "key": "~/.ssh/id_rsa"})
gpu = rh.cluster(name="my-a100", instance_type="A100:1", provider="cheapest")
gpu = rh.cluster(name="my-a10", provider="gcp", instance_type="A10:1", zone="us-west1-b", image_id="id-1332353432432", spot=True)
```

When you send something to a cluster, we check that the cluster's up (and bring it up if not), and start Ray
and a Runhouse HTTP server on it via SSH. There are lots of things you can send to the cluster. For example,
a folder, from local or cloud storage (these work in any direction, so you can send folders arbitrarily between local,
cloud storage, and cluster storage):

```python
my_cluster_folder = rh.folder(path="./my_folder").to(gpu, path="~/my_folder")
my_s3_folder = my_cluster_folder.to("s3", path="my_bucket/my_folder")
my_local_folder = my_s3_folder.to("here")
```

Or an "environment", which lives in its own Ray process. Local folders are synced up as needed, and the environment
setup is cached so it only reruns if something changes.

```python
my_env = rh.env(reqs=["torch", "diffusers", "~/code/my_other_repo"],
                setup_cmds=["source activate ~/.bash_rc"],
                env_vars={"TOKEN": "1234"},
                workdir="./")
my_env = my_env.to(gpu)
```

You can send a function to the cluster and env, where it lives inside the env's Ray process. You can send it to an
existing env, or create a new one on the fly. Note that the function is not serialized, but rather reimported on the
cluster after the local git package (`"./"`) is sent up.


```python
from diffusers import StableDiffusionPipeline

def sd_generate(prompt):
    model = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-base").to("cuda")
    return model(prompt).images

if __name__ == "__main__":
    remote_sd_generate = rh.function(sd_generate).to(gpu, env=["./", "torch", "diffusers"])
    imgs = remote_sd_generate("A hot dog made out of matcha.")
    imgs[0].show()
```

## 🙅‍♀️ Runhouse is not

* An orchestrator / scheduler (Airflow, Prefect, Metaflow)
* A distributed compute DSL (Ray, Spark, Dask)
* An ML Platform (Sagemaker, Vertex, Azure ML)
* A model registry / experiment manager / MLOps framework (MLFlow, WNB, Kubeflow)
* Hosted compute (Modal, Banana, Replicate)
* A serverless framework (Serverless, Zappa, Chalice, Apex)

## 🐣 Getting Started

Please see the [Getting Started guide](https://www.run.house/docs/tutorials/quick_start).

tldr;
```commandline
pip install runhouse
# Or "runhouse[aws]", "runhouse[gcp]", "runhouse[azure]", "runhouse[sagemaker]", "runhouse[all]"

# [optional] to set up cloud provider secrets:
sky check

# [optional] login for portability:
runhouse login
```

## 🔒 Creating a Runhouse Den Account for Secrets and Sharing

You can unlock some unique portability features by creating an (always free)
[Den account](https://www.run.house) and saving your secrets and resource metadata there.
Log in from anywhere to access all previously saved secrets and resources, ready to be used with
no additional setup.

To log in, run `runhouse login` from the command line, or
`rh.login()` from Python.

> **Note**:
Secrets are stored in Hashicorp Vault (an industry standard for secrets management), and our APIs simply call Vault's APIs. We only ever store light metadata about your resources
(e.g. my_folder_name -> [provider, bucket, path]) on our API servers, while all actual data and compute
stays inside your own cloud account and never hits our servers. We plan to
add support for BYO secrets management shortly. Let us know if you need it and which system you use.


## <h2 id="supported-infra"> 🏗️ Supported Infra </h2>

Runhouse is an ambitious project to provide a unified API into many paradigms and providers for
various types of infra. You can find our currently support systems and high-level roadmap below.
Please reach out (first name at run.house) to contribute or share feedback!
  - Single box - **Supported**
  - Ray cluster - **Supported**
  - Kubernetes - **In Progress**
  - AWS
    - EC2 - **Supported**
    - SageMaker - **Supported**
    - Lambda - **In Progress**
  - GCP - **Supported**
  - Azure - **Supported**
  - Lambda Labs - **Supported**
  - Modal - Planned
  - Slurm - Exploratory

## 👨‍🏫 Learn More

[**Docs**](https://www.run.house/docs):
Detailed API references, basic API examples and walkthroughs, end-to-end tutorials, and high-level architecture overview.

[**Funhouse Repo**](https://github.com/run-house/funhouse): Standalone Runhouse apps to try out fun ML ideas,
think the latest Stable Diffusion models, text generation models, launching Gradio spaces, and even more!

## 🙋‍♂️ Getting Help

Message us on [Discord](https://discord.gg/RnhB6589Hs), email us (first name at run.house), or create an issue.

## 👷‍♀️ Contributing

We welcome contributions! Please check out [contributing](CONTRIBUTING.md) if you're interested.
