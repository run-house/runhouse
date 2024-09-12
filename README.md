# 🏃‍♀️Runhouse🏠

[![Discord](https://dcbadge.vercel.app/api/server/RnhB6589Hs?compact=true&style=flat)](https://discord.gg/RnhB6589Hs)
[![Twitter](https://img.shields.io/twitter/url/https/twitter.com/runhouse_.svg?style=social&label=@runhouse_)](https://twitter.com/runhouse_)
[![Website](https://img.shields.io/badge/run.house-green)](https://www.run.house)
[![Docs](https://img.shields.io/badge/docs-blue)](https://www.run.house/docs)
[![Den](https://img.shields.io/badge/runhouse_den-purple)](https://www.run.house/login)

## 👵 Welcome Home!
Runhouse enables rapid, cost-effective Machine Learning development across research and production.
It allows you to dispatch Python functions and classes to any of your own cloud compute infrastructure, and call them
eagerly as if they were local. This means:
1. You can natively run and debug your code on remote GPUs or other powerful infra, like Ray, Spark, or Kubernetes,
from your laptop. Your application code then runs as-is in CI/CD or production, still dispatching work to remote infra.
2. Your application, including the infrastructure steps, is captured in code in a way that eliminates manual gruntwork
and is exactly reproducible across your team and across research and production.
3. Your flexibility to scale and cost-optimize is unmatched, with teams often seeing cost savings of ~50%.

Orchestrating across clusters, regions, or clouds is trivial, as is complex logic like scaling, fault
tolerance, or multi-step workflows.

## What is Runhouse For?
* When research-to-production is slow and painful, both due to mismatched research & production data/environments and orchestrator pipelines' lack of debuggability.
* If teams need an infra-agnostic way to execute Python to flexibly run a single workflow across heterogeneous compute, even running on multiple cloud providers.
* Ending frustration at platforms-in-a-box like SageMaker or Vertex, and moving to a more flexible solution to develop and deploy ML code.
* Growing ML maturity, as organizations move from one-off ML projects to at-scale ML flywheel.

## Highlights:
* 🚀 Dispatch Python functions, classes, and data to remote infra instantly, and call them eagerly as if they were local. Deployment/redeployment is nearly instant and logs are streamed back, making iteration extremely fast.
* 🐍 No DSL, decorators, yaml, CLI incantations, or boilerplate. Just your own regular Python, deployable to anywhere you run Python.
* 👩‍🔬 No special packaging or deployment processing is needed; research and production code are identical. Call Runhouse-deployed functions from CI/CD, Orchestrators, or applications like a micro-service.
* 👩‍🎓 BYO-infra with extensive and growing support - Ray, Kubernetes, AWS, GCP, Azure, local, on-prem, and more.
* 👩‍🚀 Extreme reproducibility and portability. There's no lock-in, because when you want to shift, scale, or pick the cheapest pricing, changing infra is as easy as changing 1 line specifying a different cluster.
* 👷‍♀️ Share Python functions or classes as robust services, including HTTPS, auth, observability, scaling, custom domains, secrets, versioning, and more.
* 👩‍🍳 Support complex workflows or services and advanced logic since your components are de-coupled and infra/modules are interactable with code.

The Runhouse API is simple. Send your **modules** (functions and classes) into **environments** (worker processes) on compute
**infra**, like this:

```python
import runhouse as rh
from diffusers import StableDiffusionPipeline

def sd_generate(prompt, **inference_kwargs):
    model = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-base").to("cuda")
    return model(prompt, **inference_kwargs).images

if __name__ == "__main__":
    gpu = rh.cluster(name="rh-a10x", instance_type="A10G:1", provider="aws").up_if_not()
    sd_env = rh.env(reqs=["torch", "transformers", "diffusers"], name="sd_env")

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


## 🛋️ Infra Monitoring, Resource Sharing and Versioning with Runhouse Den

You can unlock unique observability and sharing features with
[Runhouse Den](https://www.run.house/dashboard), a complementary product to this repo.

After you've sent a function or class to remote compute, Runhouse allows you to persist and share it as
a service, turning otherwise redundant AI activities into common modular components across your team or company.
* This makes the shared resource observable. With Den, you can see how often a resource was called (and by whom), and what was the GPU utilization of the box it was on.
* This improves cost - think 10 ML pipelines and researchers calling the same shared preprocessing, training, evaluation, or batch inference service, rather than each allocating their own compute resources
* This improves velocity and reproducibility. Avoid deploying slightly differing code per pipeline, and deploy the results of an improved method to everyone once published.

Log in from anywhere to save, share, and load resources and observe usage, logs, and compute utilization on a single pane of glass:
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

[**🐣 Getting Started**](https://www.run.house/docs/tutorials/quick-start-cloud): Installation, setup, and a quick walkthrough.

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
