<h1 align="center">🏃‍♀️ Runhouse 🏠</h1>

[//]: # (<p align="center">)

[//]: # (  <a href="https://discord.gg/RnhB6589Hs"> )

[//]: # (    <img alt="Join Discord" src="https://img.shields.io/discord/1065833240625172600?label=Discord&style=for-the-badge">)

[//]: # (  </a>)

[//]: # (</p>)

## 👵 Welcome Home!

ML infra is complicated and fragmented. Runhouse integrates with and unifies your existing ML infra,
reducing the need for any code translation or data migration, and lets you save, share, and manage your
evolving code and data artifacts.

With Runhouse,
- develop on heterogeneous hardware as if it were local, with cloud and infra agnostic, DSL-free Python APIs
- seamlessly access code and data between local, remote, and cloud
- reproducibly set up your environment and hardware, every time
- save, share, and manage living ML assets
- access production-quality features, with queuing, distributed, logging, and more

Runhouse bridges the gap between
- research and production
- local and remote
- individuals and teams

[//]: # (![img.png]&#40;docs/assets/img.png&#41;)
[//]: # (![img_1.png]&#40;docs/assets/img_1.png&#41;)
![img.png](https://raw.githubusercontent.com/run-house/runhouse/main/docs/assets/img.png)
![img_1.png](https://raw.githubusercontent.com/run-house/runhouse/main/docs/assets/img_1.png)

### Runhouse lets you do this:

On the compute side, run your local function on remote hardware, reproducibly set up your environment,
and save this information for future runs. Take the following Stable Diffusion example.

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

## 🚨 This is an Alpha 🚨
Runhouse is heavily under development and we expect to iterate on the APIs before reaching beta (version 0.1.0).

## 🐣 Getting Started

tldr;
```commandline
pip install runhouse
# Or "runhouse[aws]", "runhouse[gcp]", "runhouse[azure]", "runhouse[all]"

# [optional] to set up cloud provider secrets:
sky check

# [optional] login for portability:
runhouse login
```

### 🔌 Installation

>**Note**:
⚠️ On Apple M1 or M2 machines ⚠️ you will need to install grpcio with conda following the instructions
[here](https://docs.ray.io/en/master/ray-overview/installation.html#m1-mac-apple-silicon-support)
before you install Runhouse - more specifically, before you install Ray. If you already have Ray installed,
you can skip this. You should be able to successfully run `ray.init()` in a Python interpreter. If you're
having trouble with this, let us know.

Runhouse can be installed with:
```
pip install runhouse
```

To install cloud-provider specific dependencies for tools like boto, gsutil, etc:
```commandline
pip install "runhouse[aws]"
pip install "runhouse[gcp]"
pip install "runhouse[azure]"
pip install "runhouse[all]"
```

### Hardware and Cloud Setup

Runhouse supports both BYO (bring-your-own) cluster and on-demand, autoscaled clusters, where we spin up and
down cloud instances in your own cloud account for you.

For BYO clusters, you simply need to have the IP address and SSH key on hand.

```python
my_cluster = rh.cluster(ips=['<ip of the cluster>'],
                        ssh_creds={'ssh_user': '...', 'ssh_private_key':'<path_to_key>'},
                        name='my-byo-cluster')
```

For on-demand clusters, Runhouse uses [SkyPilot](https://skypilot.readthedocs.io/en/latest/) for
much of the heavy lifting with launching and terminating cloud instances
(we love it and you should [throw them a Github star ⭐️](https://github.com/skypilot-org/skypilot/))

To verify that your cloud credentials are set up correctly for autoscaling, run `sky check` in your
command line. This will confirm which cloud providers are set up, and give instructions for setting
up other cloud providers.

### 🔒 Creating a Runhouse Account for Secrets and Portability

You can unlock some unique portability features by creating an (always free)
account on [api.run.house](https://api.run.house) and saving your secrets and resource metadata there.
Log in from anywhere to access all previously saved secrets and resources, ready to be used with with
no additional setup.

Think of the OSS-package-only experience as akin to Microsoft Office,
while creating an account will make your cloud resources sharable and accessible
from anywhere like Google Docs.

To log in, run `runhouse login` from the command line, or
`rh.login()` from Python.

> **Note**:
Secrets are stored in Hashicorp Vault (an industry standard for secrets management), and our APIs simply call Vault's APIs. We only ever store light metadata about your resources
(e.g. my_folder_name -> [provider, bucket, path]) on our API servers, while all actual data and compute
stays inside your own cloud account and never hits our servers. We plan to
add support for BYO secrets management shortly. Let us know if you need it and which system you use.

## 👨‍🏫 Learn More

[**Docs Page**](https://runhouse-docs.readthedocs-hosted.com/en/latest/index.html):
High-level overviews of the architecture, detailed API references, and basic API examples.

[**Tutorials Repo**](https://github.com/run-house/tutorials): A comprehensive walkthrough of Runhouse APIs through some popular ML examples, think Stable Diffusion, Dreambooth, BERT.

[**Funhouse Repo**](https://github.com/run-house/funhouse): Standalone Runhouse scripts to try out fun ML ideas,
think the latest Stable Diffusion models, text generation models, launching Gradio spaces, and even more!

[**Comparisons Repo**](https://github.com/run-house/comparisons): Comparisons of Runhouse with other ML solutions, with working code examples.

## 🙋‍♂️ Getting Help

Message us on [Discord](https://discord.gg/RnhB6589Hs), email us (first name at run.house), or create an issue.

## 👷‍♀️ Contributing

We welcome contributions! Please check out [contributing](CONTRIBUTING.md) if you're interested.
