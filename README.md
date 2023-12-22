<br>

# 🏃‍♀️Runhouse🏠

[![Discord](https://dcbadge.vercel.app/api/server/RnhB6589Hs?compact=true&style=flat)](https://discord.gg/RnhB6589Hs)
[![Twitter](https://img.shields.io/twitter/url/https/twitter.com/runhouse_.svg?style=social&label=@runhouse_)](https://twitter.com/runhouse_)
[![Website](https://img.shields.io/badge/run.house-green)](https://www.run.house)
[![Docs](https://img.shields.io/badge/docs-blue)](https://www.run.house/docs)
[![Den](https://img.shields.io/badge/runhouse_den-purple)](https://www.run.house/login)

## 👵 Welcome Home!

[Runhouse](https://run.house) is a Python framework for composing and sharing production-quality backend apps and services _ridiculously
quickly_ and on your own infra. Running the following will stand up a microservice on a fresh AWS EC2 box:

```python
import runhouse as rh

def run_home(name: str):
    return f"Run home {name}!"

if __name__ == "__main__":
    cpu_box = rh.ondemand_cluster(name="my-cpu", instance_type="CPU:2", provider="aws")
    remote_fn = rh.function(run_home).to(cpu_box)
    print(remote_fn("Jack"))
```

## 🤔 Why?

Runhouse is built to do four things:
1. Make it easy to send an arbitrary block of your code - function, subroutine, class, generator -
to run on souped up remote infra. It's basically a flag flip.
1. Eliminate CLI and Flask/FastAPI boilerplate by allowing you to send your function or class directly to your remote
infra to execute or serve, and keep them debuggable like the original code, not a `subprocess.Popen` or Postman/`curl` call.
1. Bake-in the middleware and automation to make your app production-quality, secure, and sharable instantly.
That means giving you out of the box, state-of-the-art auth, HTTPS, telemetry, packaging, developer tools and deployment automation, with ample
flexibility to swap in your own.
1. Bring the power of [Ray](https://www.ray.io/) to any app, anywhere, without having to learn Ray or manage Ray
clusters, like [Next.js](https://nextjs.org/) did for React. OpenAI, Uber, Shopify, and many others use Ray to
power their ML infra, and Runhouse makes its best-in-class features accessible to any project, team, or company.

## 🤨 Who is this for?

* 👩‍🔧 **Engineers, Researchers and Data Scientists** who don't want to spend 3-6 months translating and packaging
their work to share it, and want to be able to iterate and improve production services, pipelines, experiments and data artifacts
with a Pythonic, debuggable devX.
* 👩‍🔬 **ML and data teams** who want a versioned, shared, maintainable stack of services used across
research and production, spanning any cloud or infra type (e.g. Instances, Kubernetes, Serverless, etc.).
* 🦸‍♀️ **OSS maintainers** who want to supercharge their setup flow by providing a single script to stand up their app
on any infra, rather than build support or guides for each cloud or compute system (e.g. Kubernetes) one by one.
   * See this in action in 🤗 Hugging Face ([Transformers](https://github.com/huggingface/transformers/blob/main/examples/README.md#running-the-examples-on-remote-hardware-with-auto-setup), [Accelerate](https://github.com/huggingface/accelerate/blob/main/examples/README.md#simple-multi-gpu-hardware-launcher)) and 🦜🔗 [Langchain](https://python.langchain.com/en/latest/modules/models/llms/integrations/runhouse.html)

## 🦾 How does it work?

In Runhouse, a ["cluster"](https://www.run.house/docs/api/python/cluster#cluster) is a unit of compute - somewhere you can send code, data, and requests to execute. It
can represent long-lived compute like a static IP or Ray cluster, or ephemeral/scale-to-zero compute like on-demand VMs
from a cloud provider or Kubernetes (local Docker support coming soon). When you first use a cluster, we check that the hardware is up if applicable (bringing it up automatically if not), and start Ray and a Runhouse HTTP server on it via SSH. Suppose you create a cluster object:

```python
import runhouse as rh

gpu = rh.cluster(name="my-a100", host=my_cluster_ip, ssh_creds={"ssh_user": "ubuntu", "key": "~/.ssh/id_rsa"})
gpu = rh.cluster(name="my-a100", instance_type="A100:1", provider="cheapest")
cpus = rh.cluster(name="my-cpus", provider="gcp", zone="us-west1-b", spot=True,
                 instance_type="CPU:32", disk_size="100+", memory="32+", open_ports=[8080],
                 image_id="id-1332353432432")
```

There are lots of things you can send to a cluster. For example, a [folder](https://www.run.house/docs/api/python/folder#folder), from local or cloud storage (these work
in any direction, so you can send folders arbitrarily between local, cloud storage, and cluster storage):

```python
my_cluster_folder = rh.folder(path="./my_folder").to(gpu, path="~/my_folder")
my_s3_folder = my_cluster_folder.to("s3", path="my_bucket/my_folder")
my_local_folder = my_s3_folder.to("here")
```

You can send a [function](https://www.run.house/docs/api/python/function#function) to the cluster, including the environment in which the function will live, which is actually
set up in its own Ray process. You can send it to an existing env, or create a new one on the fly. Like the folder
above, the function object which is returned from `.to` is a proxy to the remote function. When you call it, a
lightweight request is sent to the cluster's Runhouse HTTP server to execute the function with the given inputs and
returns the results. Note that the function is not serialized, but rather imported on the cluster after the local
working directory (`"./"`, by default the git root) is sent up.


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

The above env list is a shorthand, but you can create quite elaborate envs (even quite a bit more elaborate than the
example below). Local folders are sent (rsynced) up as needed, and the environment setup is cached so it only reruns
if something changes.

```python
my_env = rh.env(name="my_env",
                reqs=["torch", "diffusers", "~/code/my_other_repo"],
                setup_cmds=["source activate ~/.bash_rc"],
                secrets=["huggingface"],
                env_vars={"TOKEN": "1234"},
                workdir="./")  # Current git root
my_env = my_env.to(gpu)  # Called implicitly if passed to Function.to, like above
```

`rh.Function` is actually a special case of `rh.Module`, which is how we send classes to clusters. There are a number of
other built-in Modules in Runhouse, such as Blob, Queue, KVstore, and more. Modules are very flexible. You can
leave them as classes or create instances, send the class or the instances to clusters, and even create instances
remotely. You can call class methods, properties, generators, and async methods remotely, and they'll behave
the same as if they were called locally (e.g. generators will stream, `async`s will return `awaitable`s), including
streaming logs and stdout back to you.

```python
# You can create a module out of an existing class
MyRemoteClass = rh.module(MyClass).to(gpu)
MyRemoteClass.my_class_method(1, 2, 3)
# Notice how we sent the module to gpu above, so now this instance already lives on gpu
my_remote_instance = MyRemoteClass(1, 2, 3)

# You can define a new module as a subclass of rh.Module to have more control
# over how it's instantiated, or provide it within a library
class MyCounter(rh.Module):
    def __init__(self, count, **kwargs):
        super().__init__(**kwargs)
        self.count = count

    def increment(self, y):
        self.count += y
        print(f"New count: {self.count}")
        return self.count

my_remote_counter = MyCounter(count=1).to(gpu, name="my_counter")
my_remote_counter.increment(2)  # Prints "New count: 3" and returns 3
```

You can also call the Runhouse HTTP server directly (though you may need to open a port or tunnel to do so):
```bash
curl -X POST -H "Content-Type: application/json" http://my_cluster_address:32300/call/my_counter/count
```

This is only the tip of the iceberg. If you like what you see, please check out the
[🐣 Getting Started guide](https://www.run.house/docs/tutorials/quick_start).

## 🛋️ Creating a Runhouse Den Account for Secrets and Sharing

You can unlock some unique portability features by creating an (always free)
[Den account](https://www.run.house) and saving your secrets and resource metadata there.
Log in from anywhere to access all previously saved secrets and resources, ready to be used with
no additional setup.

To log in, run:
```shell
runhouse login
```
or from Python:
```python
import runhouse as rh
rh.login()
```

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
  - Kubernetes (K8S) - **In Progress**
  - Docker - **In Progress**
  - Amazon Web Services (AWS)
    - EC2 - **Supported**
    - SageMaker - **Supported**
    - Lambda - **Alpha**
  - Google Cloud Platform (GCP) - **Supported**
  - Microsoft Azure - **Supported**
  - Lambda Labs - **Supported**
  - Modal Labs - Planned
  - Slurm - Exploratory

## 👨‍🏫 Learn More

[**🐣 Getting Started**](https://www.run.house/docs/tutorials/quick_start): Installation, setup, and a quick walkthrough.

[**📖 Docs**](https://www.run.house/docs):
Detailed API references, basic API examples and walkthroughs, end-to-end tutorials, and high-level architecture overview.

[**🎪 Funhouse**](https://github.com/run-house/funhouse): Standalone ML apps and examples to try with Runhouse, like image generation models, LLMs,
launching Gradio spaces, and more!

[**👩‍💻 Blog**](https://www.run.house/blog): Deep dives into Runhouse features, use cases, and the future of ML
infra.

[**👾 Discord**](https://discord.gg/RnhB6589Hs): Join our community to ask questions, share ideas, and get help.

[**𝑋 Twitter**](https://twitter.com/runhouse_): Follow us for updates and announcements.

## 🙋‍♂️ Getting Help

Message us on [Discord](https://discord.gg/RnhB6589Hs), email us (first name at run.house), or create an issue.

## 👷‍♀️ Contributing

We welcome contributions! Please check out [contributing](CONTRIBUTING.md) if you're interested.
