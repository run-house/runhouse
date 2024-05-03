# 🏃‍♀️Runhouse🏠

[![Discord](https://dcbadge.vercel.app/api/server/RnhB6589Hs?compact=true&style=flat)](https://discord.gg/RnhB6589Hs)
[![Twitter](https://img.shields.io/twitter/url/https/twitter.com/runhouse_.svg?style=social&label=@runhouse_)](https://twitter.com/runhouse_)
[![Website](https://img.shields.io/badge/run.house-green)](https://www.run.house)
[![Docs](https://img.shields.io/badge/docs-blue)](https://www.run.house/docs)
[![Den](https://img.shields.io/badge/runhouse_den-purple)](https://www.run.house/login)

## 👵 Welcome Home!

Runhouse gives your code the superpower of traversing remote infrastructure, so you
can iterate and debug your ML apps and workflows locally in regular Python (no DSLs, yaml, or prescriptive
dev environment) with full-scale compute and data (no sandbox). It's the fastest way to build, run,
and deploy production-quality ML apps and workflows on your own infrastructure, and perhaps the only way to
take production code and run it as-is locally (again, running on identical powerful infra) to iterate it further or
debug.

After you've sent a function or class to remote compute, Runhouse also allows you to persist, reuse, and share it as
a service, turning otherwise redundant AI activities into common modular components across your team or company.
This improves cost, velocity, and reproducibility - think 10 ML pipelines and researchers calling the same shared
preprocessing, training, evaluation, or batch inference service, rather than each allocating their own compute
resources and deploying slightly differing code. Or, imagine experimenting with a new preprocessing method in a
notebook, but you can call every other stage of your ML workflow as the production services themselves.

Highlights:
* 👩‍🔬 Dispatch Python functions, classes, and data to remote infra instantly, and call
them eagerly as if they were local. Logs are streamed, iteration is fast.
* 👷‍♀️ Share Python functions or classes as robust services, including HTTPS, auth, observability,
scaling, custom domains, secrets, versioning, and more.
* 🐍 No DSL, decorators, yaml, CLI incantations, or boilerplate. Just your own regular Python.
* 🚀 Deploy anywhere you run Python. No special packaging or deployment process. Research and production code are
identical.
* 👩‍🎓 BYO-infra with extensive and growing support - Ray, Kubernetes, AWS, GCP, Azure, local, on-prem, and more.
When you want to shift or scale, just send your code to more powerful infra.
* 👩‍🚀 Extreme reproducibility and portability. A single succinct script can allocate the infra, set up dependencies,
and serve your app.
* 👩‍🍳 Nest applications to create complex workflows and services. Components are decoupled so you can change,
shift, or scale any component without affecting the rest of your system.

The Runhouse API is dead simple. Send your **modules** (functions and classes) into **environments** on compute
**infra**, like this:

```python
import runhouse as rh
from diffusers import StableDiffusionPipeline

def sd_generate(prompt, **inference_kwargs):
    model = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-base").to("cuda")
    return model(prompt, **inference_kwargs).images

if __name__ == "__main__":
    gpu = rh.cluster(name="rh-a10x", instance_type="A10G:1", provider="aws")
    sd_env = rh.env(reqs=["torch", "transformers", "diffusers"], name="sd_generate", working_dir="./")

    # Deploy the function and environment (syncing over local code changes and installing dependencies)
    remote_sd_generate = rh.function(sd_generate).to(gpu, env=sd_env)

    # This call is actually an HTTP request to the app running on the remote server
    imgs = remote_sd_generate("A hot dog made out of matcha.")
    imgs[0].show()

    # You can also call it over HTTP directly, e.g. from other machines or languages
    print(remote_sd_generate.endpoint())
```

With the above simple structure you can build, call, and share:
* 🛠️ **AI primitives**: Preprocessing, training, fine-tuning, evaluation, inference
* 🚀 **Higher-order services**: Multi-step inference, e2e workflows, evaluation gauntlets, HPO
* 🧪 **UAT endpoints**: Instant endpoints for client teams to test and integrate
* 🦺 **Best-practice utilities**: PII obfuscation, content moderation, data augmentation


## 🛋️ Sharing and Versioning with Runhouse Den

You can unlock unique accessibility and sharing features with
[Runhouse Den](https://www.run.house/dashboard), a complimentary product to this repo.
Log in from anywhere to save, share, and load resources:
```shell
runhouse login
```
or from Python:
```python
import runhouse as rh
rh.login()
```

Extending the example above to share and load our app via Den:

```python
remote_sd_generate.share(["my_pal@email.com"])

# The service stub can now be reloaded from anywhere, always at yours and your collaborators' fingertips
# Notice this code doesn't need to change if you update, move, or scale the service
remote_sd_generate = rh.function("/your_username/sd_generate")
imgs = remote_sd_generate("More matcha hotdogs.")
imgs[0].show()
```

## <h2 id="supported-infra"> 🏗️ Supported Compute Infra </h2>

Please reach out (first name at run.house) if you don't see your favorite compute here.
  - Local - **Supported**
  - Single box - **Supported**
  - Ray cluster - **Supported**
  - Kubernetes - **Supported**
  - Amazon Web Services (AWS)
    - EC2 - **Supported**
    - EKS - **Supported**
    - SageMaker - **Supported**
    - Lambda - **Alpha**
  - Google Cloud Platform (GCP)
    - GCE - **Supported**
    - GKE - **Supported**
  - Microsoft Azure
    - VMs - **Supported**
    - AKS - **Supported**
  - Lambda Labs - **Supported**
  - Modal Labs - Planned
  - Slurm - Exploratory

## 👨‍🏫 Learn More

[**🐣 Getting Started**](https://www.run.house/docs/tutorials/cloud_quick_start): Installation, setup, and a quick walkthrough.

[**📖 Docs**](https://www.run.house/docs):
Detailed API references, basic API examples and walkthroughs, end-to-end tutorials, and high-level architecture overview.

[**👩‍💻 Blog**](https://www.run.house/blog): Deep dives into Runhouse features, use cases, and the future of AI
infra.

[**👾 Discord**](https://discord.gg/RnhB6589Hs): Join our community to ask questions, share ideas, and get help.

[**𝑋 Twitter**](https://twitter.com/runhouse_): Follow us for updates and announcements.

## 🙋‍♂️ Getting Help

Message us on [Discord](https://discord.gg/RnhB6589Hs), email us (first name at run.house), or create an issue.

## 👷‍♀️ Contributing

We welcome contributions! Please check out [contributing](CONTRIBUTING.md).
