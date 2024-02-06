Inference: Stable Diffusion and FLAN-T5
=======================================

.. raw:: html

    <p><a href="https://colab.research.google.com/github/run-house/runhouse/blob/stable/docs/notebooks/examples/inference.ipynb">
    <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a></p>

Runhouse is nothing more than an accessibility and sharing layer into
your own cloud compute and data resources. This tutorial demonstrates
how to create, setup, and run remote functions on **your own GPU**,
using Stable Diffusion inference as an example.

You can run this on your own cluster, or through a standard cloud
account (AWS, GCP, Azure, LambdaLabs). If you do not have any compute or
cloud accounts set up, we recommend creating a
`LambdaLabs <https://cloud.lambdalabs.com/>`__ account for the easiest
setup path.

Table of Contents
-----------------

-  Hardware Setup
-  Stable Diffusion on a Cloud GPU in 5 lines of code
-  Faster Stable Diffusion
-  FLAN-T5 Stable Diffusion

Hardware Setup
--------------

Install Runhouse
~~~~~~~~~~~~~~~~

.. code:: ipython3

    !pip install runhouse[sky]

.. code:: ipython3

    import runhouse as rh


.. parsed-literal::
    :class: code-output

    INFO | 2023-02-07 15:37:56,202 | Loaded Runhouse config from /root/.rh/config.yaml
    INFO | 2023-02-07 15:37:56,965 | NumExpr defaulting to 2 threads.


[Optional] Login to Runhouse to load in secrets
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    rh.login(download_secrets=True, download_config=True, interactive=True)



.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">
                ____              __                             @ @ @
               <span style="color: #800080; text-decoration-color: #800080">/</span> __ \__  ______  <span style="color: #800080; text-decoration-color: #800080">/</span> <span style="color: #800080; text-decoration-color: #800080">/</span><span style="color: #ff00ff; text-decoration-color: #ff00ff">_</span>  ____  __  __________     <span style="font-weight: bold">[]</span>___
              <span style="color: #800080; text-decoration-color: #800080">/</span> <span style="color: #800080; text-decoration-color: #800080">/_/</span> <span style="color: #800080; text-decoration-color: #800080">/</span> <span style="color: #800080; text-decoration-color: #800080">/</span> <span style="color: #800080; text-decoration-color: #800080">/</span> <span style="color: #800080; text-decoration-color: #800080">/</span> __ \<span style="color: #800080; text-decoration-color: #800080">/</span> __ \<span style="color: #800080; text-decoration-color: #800080">/</span> __ \<span style="color: #800080; text-decoration-color: #800080">/</span> <span style="color: #800080; text-decoration-color: #800080">/</span> <span style="color: #800080; text-decoration-color: #800080">/</span> <span style="color: #800080; text-decoration-color: #800080">/</span> ___/ _ \   <span style="color: #800080; text-decoration-color: #800080">/</span>    <span style="color: #800080; text-decoration-color: #800080">/</span>\____    @@
             <span style="color: #800080; text-decoration-color: #800080">/</span> _, _/ <span style="color: #800080; text-decoration-color: #800080">/_/</span> <span style="color: #800080; text-decoration-color: #800080">/</span> <span style="color: #800080; text-decoration-color: #800080">/</span> <span style="color: #800080; text-decoration-color: #800080">/</span> <span style="color: #800080; text-decoration-color: #800080">/</span> <span style="color: #800080; text-decoration-color: #800080">/</span> <span style="color: #800080; text-decoration-color: #800080">/</span> <span style="color: #800080; text-decoration-color: #800080">/</span> <span style="color: #800080; text-decoration-color: #800080">/_/</span> <span style="color: #800080; text-decoration-color: #800080">/</span> <span style="color: #800080; text-decoration-color: #800080">/_/</span> <span style="font-weight: bold">(</span>__  <span style="font-weight: bold">)</span>  __/  <span style="color: #800080; text-decoration-color: #800080">/_/</span>\_/<span style="color: #800080; text-decoration-color: #800080">/____/</span>\  @@@@
            <span style="color: #800080; text-decoration-color: #800080">/_/</span> |_|\__,_/_/ <span style="color: #800080; text-decoration-color: #800080">/_/_/</span> <span style="color: #800080; text-decoration-color: #800080">/_/</span>\____/\__,_/____/\___/   | || |||__|||   ||

    </pre>




.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #808000; text-decoration-color: #808000; font-weight: bold">Retrieve your token 🔑 here to use 🏃 🏠 Runhouse for secrets and artifact management: </span><a href="https://run.house/account#token" target="_blank"><span style="color: #0000ff; text-decoration-color: #0000ff; text-decoration: underline">https://run.house/account#token</span></a>
    </pre>



.. parsed-literal::
    :class: code-output

    Token: ··········
    Upload your local config to Runhouse? [y/N]: y
    Upload your enabled cloud provider secrets to Vault? [y/N]: y
    INFO | 2023-02-07 15:39:28,273 | Getting secrets from Vault.
    INFO | 2023-02-07 15:39:29,896 | Found credentials in shared credentials file: ~/.aws/credentials
    INFO | 2023-02-07 15:39:30,765 | Saved secrets from Vault to local config files for providers: ['aws', 'sky']
    INFO | 2023-02-07 15:39:32,168 | Found credentials in shared credentials file: ~/.aws/credentials
    Upload secrets for aws? [y/N]: y
    Upload secrets for sky? [y/N]: y
    INFO | 2023-02-07 15:39:35,962 | Uploaded secrets for providers ['aws', 'sky'] to Vault
    INFO | 2023-02-07 15:39:35,965 | Successfully logged into Runhouse.


Set up Runhouse cluster
~~~~~~~~~~~~~~~~~~~~~~~

Runhouse leverages SkyPilot for cloud-provider on-demand clusters. If
using an on-demand AWS, GCP, Azure, LambdaLabs cluster, run the
``sky check`` CLI command for instructions on how to set up local
credentials for your specified cloud provider(s). You can rerun this
command after setup to check that it has been set up correctly.

.. code:: ipython3

    !sky check

.. code:: ipython3

    # Uncomment if you're using GCP and running inside Colab!
    # !gcloud init
    # !gcloud auth application-default login
    # !cp -r /content/.config/* ~/.config/gcloud

Depending on your cloud provider, uncomment out one of the following
options to instantiate your ``rh-a10x``, and save it to your rh config.

.. code:: ipython3

    # For GCP, Azure, or Lambda Labs
    # rh.ondemand_cluster(name='rh-a10x', instance_type='A100:1').save()

    # For AWS (single A100s not available, base A10G may have insufficient CPU RAM)
    # rh.ondemand_cluster(name='rh-a10x', instance_type='g5.2xlarge', provider='aws').save()

    # To use our own GPU (or from a different provider, e.g. Paperspace, Coreweave)
    # rh.cluster(ips=['<ip of the cluster>'],
    #            ssh_creds={'ssh_user': '...', 'ssh_private_key':'<path_to_key>'},
    #            name='rh-a10x').save()

Now we’re ready to get started running!

Stable Diffusion on a Cloud GPU in 5 lines of code
-----------------------------------------------------

We’ll use Runhouse to experiment with Stable Diffusion from your laptop,
while the model actually runs on an A100/A10G in the cloud.

.. code:: ipython3

    gpu = rh.cluster(name='rh-a10x')

.. code:: ipython3

    def sd_generate(prompt, num_images=1, steps=100, guidance_scale=7.5, model_id='stabilityai/stable-diffusion-2-base'):
        # imports must be defined inside the function for notebook environments
        from diffusers import StableDiffusionPipeline
        import torch

        pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16, revision='fp16').to('cuda')
        return pipe([prompt] * num_images, num_inference_steps=steps, guidance_scale=guidance_scale).images

.. code:: ipython3

    generate_gpu = rh.function(fn=sd_generate).to(
        gpu,
        env=['./', 'torch --upgrade --extra-index-url https://download.pytorch.org/whl/cu117', 'diffusers', 'transformers']
    )

.. code:: ipython3

    # for outputting images later
    !pip install ipyplot
    import ipyplot

.. code:: ipython3

    rh_prompt = 'A digital illustration of a woman running on the roof of a house.'
    images = generate_gpu(rh_prompt, num_images=4, steps=50)

    ipyplot.plot_images(images)


.. parsed-literal::
    :class: code-output

    INFO | 2023-02-07 16:10:42,374 | Running sd_generate via gRPC
    INFO | 2023-02-07 16:11:27,874 | Time to send message: 45.5 seconds
    WARNING | 2023-02-07 16:11:27,885 | /usr/local/lib/python3.8/dist-packages/ipyplot/_utils.py:95: FutureWarning: The input object of type 'Image' is an array-like implementing one of the corresponding protocols (`__array__`, `__array_interface__` or `__array_struct__`); but not a sequence (or 0-D). In the future, this object will be coerced as if it was first converted using `np.array(obj)`. To retain the old behaviour, you have to either modify the type 'Image', or assign to an empty array created with `np.empty(correct_shape, dtype=object)`.
      return np.asarray(seq, dtype=type(seq[0]))




.. raw:: html


    <style>
        #ipyplot-html-viewer-toggle-WhhNeRcdRs43RrEUjLWLiB {
            position: absolute;
            top: -9999px;
            left: -9999px;
            visibility: hidden;
        }

        #ipyplot-html-viewer-label-WhhNeRcdRs43RrEUjLWLiB {
            position: relative;
            display: inline-block;
            cursor: pointer;
            color: blue;
            text-decoration: underline;
        }

        #ipyplot-html-viewer-textarea-WhhNeRcdRs43RrEUjLWLiB {
            background: lightgrey;
            width: 100%;
            height: 0px;
            display: none;
        }

        #ipyplot-html-viewer-toggle-WhhNeRcdRs43RrEUjLWLiB:checked ~ #ipyplot-html-viewer-textarea-WhhNeRcdRs43RrEUjLWLiB {
            height: 200px;
            display: block;
        }

        #ipyplot-html-viewer-toggle-WhhNeRcdRs43RrEUjLWLiB:checked + #ipyplot-html-viewer-label-WhhNeRcdRs43RrEUjLWLiB:after {
            content: "hide html";
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: white;
            cursor: pointer;
            color: blue;
            text-decoration: underline;
        }
    </style>
    <div>
        <input type="checkbox" id="ipyplot-html-viewer-toggle-WhhNeRcdRs43RrEUjLWLiB">
        <label id="ipyplot-html-viewer-label-WhhNeRcdRs43RrEUjLWLiB" for="ipyplot-html-viewer-toggle-WhhNeRcdRs43RrEUjLWLiB">show html</label>
        <textarea id="ipyplot-html-viewer-textarea-WhhNeRcdRs43RrEUjLWLiB" readonly>

        <style>
        #ipyplot-imgs-container-div-ciwsTsagCbGynZrGzcK5cc {
            width: 100%;
            height: 100%;
            margin: 0%;
            overflow: auto;
            position: relative;
            overflow-y: scroll;
        }

        div.ipyplot-placeholder-div-ciwsTsagCbGynZrGzcK5cc {
            width: 150px;
            display: inline-block;
            margin: 3px;
            position: relative;
        }

        div.ipyplot-content-div-ciwsTsagCbGynZrGzcK5cc {
            width: 150px;
            background: white;
            display: inline-block;
            vertical-align: top;
            text-align: center;
            position: relative;
            border: 2px solid #ddd;
            top: 0;
            left: 0;
        }

        div.ipyplot-content-div-ciwsTsagCbGynZrGzcK5cc span.ipyplot-img-close {
            display: none;
        }

        div.ipyplot-content-div-ciwsTsagCbGynZrGzcK5cc span {
            width: 100%;
            height: 100%;
            position: absolute;
            top: 0;
            left: 0;
        }

        div.ipyplot-content-div-ciwsTsagCbGynZrGzcK5cc img {
            width: 150px;
        }

        div.ipyplot-content-div-ciwsTsagCbGynZrGzcK5cc span.ipyplot-img-close:hover {
            cursor: zoom-out;
        }
        div.ipyplot-content-div-ciwsTsagCbGynZrGzcK5cc span.ipyplot-img-expand:hover {
            cursor: zoom-in;
        }

        div[id^=ipyplot-content-div-ciwsTsagCbGynZrGzcK5cc]:target {
            transform: scale(2.5);
            transform-origin: left top;
            z-index: 5000;
            top: 0;
            left: 0;
            position: absolute;
        }

        div[id^=ipyplot-content-div-ciwsTsagCbGynZrGzcK5cc]:target span.ipyplot-img-close {
            display: block;
        }

        div[id^=ipyplot-content-div-ciwsTsagCbGynZrGzcK5cc]:target span.ipyplot-img-expand {
            display: none;
        }
        </style>
    <div id="ipyplot-imgs-container-div-ciwsTsagCbGynZrGzcK5cc">
    <div class="ipyplot-placeholder-div-ciwsTsagCbGynZrGzcK5cc">
        <div id="ipyplot-content-div-ciwsTsagCbGynZrGzcK5cc-DXJFBL2Rm6vZ4T4kcCvGyv" class="ipyplot-content-div-ciwsTsagCbGynZrGzcK5cc">
            <h4 style="font-size: 12px; word-wrap: break-word;">0</h4>
            <img src="https://runhouse-tutorials.s3.amazonaws.com/sd_0.png" alt="digital illustration of a woman running on the roof of a house"/>
            <a href="#!">
                <span class="ipyplot-img-close"/>
            </a>
            <a href="#ipyplot-content-div-ciwsTsagCbGynZrGzcK5cc-DXJFBL2Rm6vZ4T4kcCvGyv">
                <span class="ipyplot-img-expand"/>
            </a>
        </div>
    </div>

    <div class="ipyplot-placeholder-div-ciwsTsagCbGynZrGzcK5cc">
        <div id="ipyplot-content-div-ciwsTsagCbGynZrGzcK5cc-MoGxsQQg5t2VXMZnYGaNqD" class="ipyplot-content-div-ciwsTsagCbGynZrGzcK5cc">
            <h4 style="font-size: 12px; word-wrap: break-word;">1</h4>
            <img src="https://runhouse-tutorials.s3.amazonaws.com/sd_1.png" alt="digital illustration of a woman running on the roof of a house"/>
            <a href="#!">
                <span class="ipyplot-img-close"/>
            </a>
            <a href="#ipyplot-content-div-ciwsTsagCbGynZrGzcK5cc-MoGxsQQg5t2VXMZnYGaNqD">
                <span class="ipyplot-img-expand"/>
            </a>
        </div>
    </div>

    <div class="ipyplot-placeholder-div-ciwsTsagCbGynZrGzcK5cc">
        <div id="ipyplot-content-div-ciwsTsagCbGynZrGzcK5cc-H8bdZVYWHTKpowowURxzdu" class="ipyplot-content-div-ciwsTsagCbGynZrGzcK5cc">
            <h4 style="font-size: 12px; word-wrap: break-word;">2</h4>
            <img src="https://runhouse-tutorials.s3.amazonaws.com/sd_2.png" alt="digital illustration of a woman running on the roof of a house"/>
            <a href="#!">
                <span class="ipyplot-img-close"/>
            </a>
            <a href="#ipyplot-content-div-ciwsTsagCbGynZrGzcK5cc-H8bdZVYWHTKpowowURxzdu">
                <span class="ipyplot-img-expand"/>
            </a>
        </div>
    </div>

    <div class="ipyplot-placeholder-div-ciwsTsagCbGynZrGzcK5cc">
        <div id="ipyplot-content-div-ciwsTsagCbGynZrGzcK5cc-idzjZHiuzvWESgu9PPssMx" class="ipyplot-content-div-ciwsTsagCbGynZrGzcK5cc">
            <h4 style="font-size: 12px; word-wrap: break-word;">3</h4>
            <img src="https://runhouse-tutorials.s3.amazonaws.com/sd_3.png" alt="digital illustration of a woman running on the roof of a house"/>
            <a href="#!">
                <span class="ipyplot-img-close"/>
            </a>
            <a href="#ipyplot-content-div-ciwsTsagCbGynZrGzcK5cc-idzjZHiuzvWESgu9PPssMx">
                <span class="ipyplot-img-expand"/>
            </a>
        </div>
    </div>
    </div>
        </textarea>
    </div>




.. raw:: html


        <style>
        #ipyplot-imgs-container-div-ciwsTsagCbGynZrGzcK5cc {
            width: 100%;
            height: 100%;
            margin: 0%;
            overflow: auto;
            position: relative;
            overflow-y: scroll;
        }

        div.ipyplot-placeholder-div-ciwsTsagCbGynZrGzcK5cc {
            width: 150px;
            display: inline-block;
            margin: 3px;
            position: relative;
        }

        div.ipyplot-content-div-ciwsTsagCbGynZrGzcK5cc {
            width: 150px;
            background: white;
            display: inline-block;
            vertical-align: top;
            text-align: center;
            position: relative;
            border: 2px solid #ddd;
            top: 0;
            left: 0;
        }

        div.ipyplot-content-div-ciwsTsagCbGynZrGzcK5cc span.ipyplot-img-close {
            display: none;
        }

        div.ipyplot-content-div-ciwsTsagCbGynZrGzcK5cc span {
            width: 100%;
            height: 100%;
            position: absolute;
            top: 0;
            left: 0;
        }

        div.ipyplot-content-div-ciwsTsagCbGynZrGzcK5cc img {
            width: 150px;
        }

        div.ipyplot-content-div-ciwsTsagCbGynZrGzcK5cc span.ipyplot-img-close:hover {
            cursor: zoom-out;
        }
        div.ipyplot-content-div-ciwsTsagCbGynZrGzcK5cc span.ipyplot-img-expand:hover {
            cursor: zoom-in;
        }

        div[id^=ipyplot-content-div-ciwsTsagCbGynZrGzcK5cc]:target {
            transform: scale(2.5);
            transform-origin: left top;
            z-index: 5000;
            top: 0;
            left: 0;
            position: absolute;
        }

        div[id^=ipyplot-content-div-ciwsTsagCbGynZrGzcK5cc]:target span.ipyplot-img-close {
            display: block;
        }

        div[id^=ipyplot-content-div-ciwsTsagCbGynZrGzcK5cc]:target span.ipyplot-img-expand {
            display: none;
        }
        </style>
    <div id="ipyplot-imgs-container-div-ciwsTsagCbGynZrGzcK5cc">
    <div class="ipyplot-placeholder-div-ciwsTsagCbGynZrGzcK5cc">
        <div id="ipyplot-content-div-ciwsTsagCbGynZrGzcK5cc-DXJFBL2Rm6vZ4T4kcCvGyv" class="ipyplot-content-div-ciwsTsagCbGynZrGzcK5cc">
            <h4 style="font-size: 12px; word-wrap: break-word;">0</h4>
            <img src="https://runhouse-tutorials.s3.amazonaws.com/sd_0.png" alt="digital illustration of a woman running on the roof of a house"/>
            <a href="#!">
                <span class="ipyplot-img-close"/>
            </a>
            <a href="#ipyplot-content-div-ciwsTsagCbGynZrGzcK5cc-DXJFBL2Rm6vZ4T4kcCvGyv">
                <span class="ipyplot-img-expand"/>
            </a>
        </div>
    </div>

    <div class="ipyplot-placeholder-div-ciwsTsagCbGynZrGzcK5cc">
        <div id="ipyplot-content-div-ciwsTsagCbGynZrGzcK5cc-MoGxsQQg5t2VXMZnYGaNqD" class="ipyplot-content-div-ciwsTsagCbGynZrGzcK5cc">
            <h4 style="font-size: 12px; word-wrap: break-word;">1</h4>
            <img src="https://runhouse-tutorials.s3.amazonaws.com/sd_1.png" alt="digital illustration of a woman running on the roof of a house"/>
            <a href="#!">
                <span class="ipyplot-img-close"/>
            </a>
            <a href="#ipyplot-content-div-ciwsTsagCbGynZrGzcK5cc-MoGxsQQg5t2VXMZnYGaNqD">
                <span class="ipyplot-img-expand"/>
            </a>
        </div>
    </div>

    <div class="ipyplot-placeholder-div-ciwsTsagCbGynZrGzcK5cc">
        <div id="ipyplot-content-div-ciwsTsagCbGynZrGzcK5cc-H8bdZVYWHTKpowowURxzdu" class="ipyplot-content-div-ciwsTsagCbGynZrGzcK5cc">
            <h4 style="font-size: 12px; word-wrap: break-word;">2</h4>
            <img src="https://runhouse-tutorials.s3.amazonaws.com/sd_2.png" alt="digital illustration of a woman running on the roof of a house"/>
            <a href="#!">
                <span class="ipyplot-img-close"/>
            </a>
            <a href="#ipyplot-content-div-ciwsTsagCbGynZrGzcK5cc-H8bdZVYWHTKpowowURxzdu">
                <span class="ipyplot-img-expand"/>
            </a>
        </div>
    </div>

    <div class="ipyplot-placeholder-div-ciwsTsagCbGynZrGzcK5cc">
        <div id="ipyplot-content-div-ciwsTsagCbGynZrGzcK5cc-idzjZHiuzvWESgu9PPssMx" class="ipyplot-content-div-ciwsTsagCbGynZrGzcK5cc">
            <h4 style="font-size: 12px; word-wrap: break-word;">3</h4>
            <img src="https://runhouse-tutorials.s3.amazonaws.com/sd_3.png" alt="digital illustration of a woman running on the roof of a house"/>
            <a href="#!">
                <span class="ipyplot-img-close"/>
            </a>
            <a href="#ipyplot-content-div-ciwsTsagCbGynZrGzcK5cc-idzjZHiuzvWESgu9PPssMx">
                <span class="ipyplot-img-expand"/>
            </a>
        </div>
    </div>
    </div>


.. code:: ipython3

    # save the function to be reusable later on
    generate_gpu.save(name='sd_generate')

.. code:: ipython3

    # By default, the GPU will terminate after 30 min of inactivity.
    # To keep it up to reuse it for the rest of the tutorials
    gpu.keep_warm()

    # To stop the cluster after 10 min of inactivity
    # gpu.keep_warm(autostop_mins=10)

    # To terminate the cluster through runhouse. It can also be terminated directly through the cloud provider
    # or by using the CLI commands `sky down gpu` or `sky down --all`
    # gpu.teardown()

Faster Stable Diffusion
-----------------------

The previous function will load the pretrained model every time the
function is run. In this section, we demonstrate two ways to reuse the
loaded model on the GPU to bring down the time to run Stable Diffusion.

Load/Retrieve from Object Store
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The first approach uses ``rh.here.put()`` and ``rh.here.get()`` to save
down and later retrieve the model from your Runhouse object store. The
model will be still need to be loaded the first run to be put in memory,
so speed-ups will only be observed in future runs.

.. code:: ipython3

    def sd_generate_pinned(prompt, num_images=1, steps=100, guidance_scale=7.5,
                           model_id='stabilityai/stable-diffusion-2-base',
                           revision="fp16"):
        import torch
        import runhouse as rh
        from diffusers import StableDiffusionPipeline, DDIMScheduler

        pipe = rh.get_pinned_object(model_id)

        # pin to memory if it is not in memory yet
        if pipe is None:
            pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16, revision=revision).to("cuda")
            pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
            rh.pin_to_memory(model_id, pipe)

        return pipe(prompt, num_images_per_prompt=num_images,
                    num_inference_steps=steps, guidance_scale=guidance_scale).images

.. code:: ipython3

    generate_pinned_gpu = rh.function(fn=sd_generate_pinned).to(gpu).save("sd_generate_pinned")
    my_prompt = 'A hot dog made of matcha powder.'
    matcha_images = generate_pinned_gpu(my_prompt, num_images=4, steps=50)

.. code:: ipython3

    ipyplot.plot_images(matcha_images)


.. parsed-literal::
    :class: code-output

    WARNING | 2023-02-07 16:29:32,700 | /usr/local/lib/python3.8/dist-packages/ipyplot/_utils.py:95: FutureWarning: The input object of type 'Image' is an array-like implementing one of the corresponding protocols (`__array__`, `__array_interface__` or `__array_struct__`); but not a sequence (or 0-D). In the future, this object will be coerced as if it was first converted using `np.array(obj)`. To retain the old behaviour, you have to either modify the type 'Image', or assign to an empty array created with `np.empty(correct_shape, dtype=object)`.
      return np.asarray(seq, dtype=type(seq[0]))




.. raw:: html


    <style>
        #ipyplot-html-viewer-toggle-SdbGRnYFKgEsQ8iX6tTzJn {
            position: absolute;
            top: -9999px;
            left: -9999px;
            visibility: hidden;
        }

        #ipyplot-html-viewer-label-SdbGRnYFKgEsQ8iX6tTzJn {
            position: relative;
            display: inline-block;
            cursor: pointer;
            color: blue;
            text-decoration: underline;
        }

        #ipyplot-html-viewer-textarea-SdbGRnYFKgEsQ8iX6tTzJn {
            background: lightgrey;
            width: 100%;
            height: 0px;
            display: none;
        }

        #ipyplot-html-viewer-toggle-SdbGRnYFKgEsQ8iX6tTzJn:checked ~ #ipyplot-html-viewer-textarea-SdbGRnYFKgEsQ8iX6tTzJn {
            height: 200px;
            display: block;
        }

        #ipyplot-html-viewer-toggle-SdbGRnYFKgEsQ8iX6tTzJn:checked + #ipyplot-html-viewer-label-SdbGRnYFKgEsQ8iX6tTzJn:after {
            content: "hide html";
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: white;
            cursor: pointer;
            color: blue;
            text-decoration: underline;
        }
    </style>
    <div>
        <input type="checkbox" id="ipyplot-html-viewer-toggle-SdbGRnYFKgEsQ8iX6tTzJn">
        <label id="ipyplot-html-viewer-label-SdbGRnYFKgEsQ8iX6tTzJn" for="ipyplot-html-viewer-toggle-SdbGRnYFKgEsQ8iX6tTzJn">show html</label>
        <textarea id="ipyplot-html-viewer-textarea-SdbGRnYFKgEsQ8iX6tTzJn" readonly>

        <style>
        #ipyplot-imgs-container-div-7j3Fk4SG4Jwxoafp6tRuiB {
            width: 100%;
            height: 100%;
            margin: 0%;
            overflow: auto;
            position: relative;
            overflow-y: scroll;
        }

        div.ipyplot-placeholder-div-7j3Fk4SG4Jwxoafp6tRuiB {
            width: 150px;
            display: inline-block;
            margin: 3px;
            position: relative;
        }

        div.ipyplot-content-div-7j3Fk4SG4Jwxoafp6tRuiB {
            width: 150px;
            background: white;
            display: inline-block;
            vertical-align: top;
            text-align: center;
            position: relative;
            border: 2px solid #ddd;
            top: 0;
            left: 0;
        }

        div.ipyplot-content-div-7j3Fk4SG4Jwxoafp6tRuiB span.ipyplot-img-close {
            display: none;
        }

        div.ipyplot-content-div-7j3Fk4SG4Jwxoafp6tRuiB span {
            width: 100%;
            height: 100%;
            position: absolute;
            top: 0;
            left: 0;
        }

        div.ipyplot-content-div-7j3Fk4SG4Jwxoafp6tRuiB img {
            width: 150px;
        }

        div.ipyplot-content-div-7j3Fk4SG4Jwxoafp6tRuiB span.ipyplot-img-close:hover {
            cursor: zoom-out;
        }
        div.ipyplot-content-div-7j3Fk4SG4Jwxoafp6tRuiB span.ipyplot-img-expand:hover {
            cursor: zoom-in;
        }

        div[id^=ipyplot-content-div-7j3Fk4SG4Jwxoafp6tRuiB]:target {
            transform: scale(2.5);
            transform-origin: left top;
            z-index: 5000;
            top: 0;
            left: 0;
            position: absolute;
        }

        div[id^=ipyplot-content-div-7j3Fk4SG4Jwxoafp6tRuiB]:target span.ipyplot-img-close {
            display: block;
        }

        div[id^=ipyplot-content-div-7j3Fk4SG4Jwxoafp6tRuiB]:target span.ipyplot-img-expand {
            display: none;
        }
        </style>
    <div id="ipyplot-imgs-container-div-7j3Fk4SG4Jwxoafp6tRuiB">
    <div class="ipyplot-placeholder-div-7j3Fk4SG4Jwxoafp6tRuiB">
        <div id="ipyplot-content-div-7j3Fk4SG4Jwxoafp6tRuiB-5aCbbqqDJtgDsv3xRXWWr8" class="ipyplot-content-div-7j3Fk4SG4Jwxoafp6tRuiB">
            <h4 style="font-size: 12px; word-wrap: break-word;">0</h4>
            <img src="https://runhouse-tutorials.s3.amazonaws.com/pin_0.png" alt="image of a prepared hot dog with green powder on top and side dishes" />
            <a href="#!">
                <span class="ipyplot-img-close"/>
            </a>
            <a href="#ipyplot-content-div-7j3Fk4SG4Jwxoafp6tRuiB-5aCbbqqDJtgDsv3xRXWWr8">
                <span class="ipyplot-img-expand"/>
            </a>
        </div>
    </div>

    <div class="ipyplot-placeholder-div-7j3Fk4SG4Jwxoafp6tRuiB">
        <div id="ipyplot-content-div-7j3Fk4SG4Jwxoafp6tRuiB-89y7xSULgpfyqd9QX8X9sf" class="ipyplot-content-div-7j3Fk4SG4Jwxoafp6tRuiB">
            <h4 style="font-size: 12px; word-wrap: break-word;">1</h4>
            <img src="https://runhouse-tutorials.s3.amazonaws.com/pin_1.png" alt="image of a bright green hot dog on a irregular-shaped plate"/>
            <a href="#!">
                <span class="ipyplot-img-close"/>
            </a>
            <a href="#ipyplot-content-div-7j3Fk4SG4Jwxoafp6tRuiB-89y7xSULgpfyqd9QX8X9sf">
                <span class="ipyplot-img-expand"/>
            </a>
        </div>
    </div>

    <div class="ipyplot-placeholder-div-7j3Fk4SG4Jwxoafp6tRuiB">
        <div id="ipyplot-content-div-7j3Fk4SG4Jwxoafp6tRuiB-XDXTQZXDutnkP2UeJ89UQE" class="ipyplot-content-div-7j3Fk4SG4Jwxoafp6tRuiB">
            <h4 style="font-size: 12px; word-wrap: break-word;">2</h4>
            <img src="https://runhouse-tutorials.s3.amazonaws.com/pin_2.png" alt="image of a hot dog in a bun with matcha topping on a wooden surface"/>
            <a href="#!">
                <span class="ipyplot-img-close"/>
            </a>
            <a href="#ipyplot-content-div-7j3Fk4SG4Jwxoafp6tRuiB-XDXTQZXDutnkP2UeJ89UQE">
                <span class="ipyplot-img-expand"/>
            </a>
        </div>
    </div>

    <div class="ipyplot-placeholder-div-7j3Fk4SG4Jwxoafp6tRuiB">
        <div id="ipyplot-content-div-7j3Fk4SG4Jwxoafp6tRuiB-g5ofaCEvZ4UHffLp3H6evR" class="ipyplot-content-div-7j3Fk4SG4Jwxoafp6tRuiB">
            <h4 style="font-size: 12px; word-wrap: break-word;">3</h4>
            <img src="https://runhouse-tutorials.s3.amazonaws.com/pin_3.png" alt="image of a hot dog in a bun spread with matcha on a marble surface"/>
            <a href="#!">
                <span class="ipyplot-img-close"/>
            </a>
            <a href="#ipyplot-content-div-7j3Fk4SG4Jwxoafp6tRuiB-g5ofaCEvZ4UHffLp3H6evR">
                <span class="ipyplot-img-expand"/>
            </a>
        </div>
    </div>
    </div>
        </textarea>
    </div>




.. raw:: html


        <style>
        #ipyplot-imgs-container-div-7j3Fk4SG4Jwxoafp6tRuiB {
            width: 100%;
            height: 100%;
            margin: 0%;
            overflow: auto;
            position: relative;
            overflow-y: scroll;
        }

        div.ipyplot-placeholder-div-7j3Fk4SG4Jwxoafp6tRuiB {
            width: 150px;
            display: inline-block;
            margin: 3px;
            position: relative;
        }

        div.ipyplot-content-div-7j3Fk4SG4Jwxoafp6tRuiB {
            width: 150px;
            background: white;
            display: inline-block;
            vertical-align: top;
            text-align: center;
            position: relative;
            border: 2px solid #ddd;
            top: 0;
            left: 0;
        }

        div.ipyplot-content-div-7j3Fk4SG4Jwxoafp6tRuiB span.ipyplot-img-close {
            display: none;
        }

        div.ipyplot-content-div-7j3Fk4SG4Jwxoafp6tRuiB span {
            width: 100%;
            height: 100%;
            position: absolute;
            top: 0;
            left: 0;
        }

        div.ipyplot-content-div-7j3Fk4SG4Jwxoafp6tRuiB img {
            width: 150px;
        }

        div.ipyplot-content-div-7j3Fk4SG4Jwxoafp6tRuiB span.ipyplot-img-close:hover {
            cursor: zoom-out;
        }
        div.ipyplot-content-div-7j3Fk4SG4Jwxoafp6tRuiB span.ipyplot-img-expand:hover {
            cursor: zoom-in;
        }

        div[id^=ipyplot-content-div-7j3Fk4SG4Jwxoafp6tRuiB]:target {
            transform: scale(2.5);
            transform-origin: left top;
            z-index: 5000;
            top: 0;
            left: 0;
            position: absolute;
        }

        div[id^=ipyplot-content-div-7j3Fk4SG4Jwxoafp6tRuiB]:target span.ipyplot-img-close {
            display: block;
        }

        div[id^=ipyplot-content-div-7j3Fk4SG4Jwxoafp6tRuiB]:target span.ipyplot-img-expand {
            display: none;
        }
        </style>
    <div id="ipyplot-imgs-container-div-7j3Fk4SG4Jwxoafp6tRuiB">
    <div class="ipyplot-placeholder-div-7j3Fk4SG4Jwxoafp6tRuiB">
        <div id="ipyplot-content-div-7j3Fk4SG4Jwxoafp6tRuiB-5aCbbqqDJtgDsv3xRXWWr8" class="ipyplot-content-div-7j3Fk4SG4Jwxoafp6tRuiB">
            <h4 style="font-size: 12px; word-wrap: break-word;">0</h4>
            <img src="https://runhouse-tutorials.s3.amazonaws.com/pin_0.png" alt="image of a prepared hot dog with green powder on top and side dishes"/>
            <a href="#!">
                <span class="ipyplot-img-close"/>
            </a>
            <a href="#ipyplot-content-div-7j3Fk4SG4Jwxoafp6tRuiB-5aCbbqqDJtgDsv3xRXWWr8">
                <span class="ipyplot-img-expand"/>
            </a>
        </div>
    </div>

    <div class="ipyplot-placeholder-div-7j3Fk4SG4Jwxoafp6tRuiB">
        <div id="ipyplot-content-div-7j3Fk4SG4Jwxoafp6tRuiB-89y7xSULgpfyqd9QX8X9sf" class="ipyplot-content-div-7j3Fk4SG4Jwxoafp6tRuiB">
            <h4 style="font-size: 12px; word-wrap: break-word;">1</h4>
            <img src="https://runhouse-tutorials.s3.amazonaws.com/pin_1.png" alt="image of a bright green hot dog on a irregular-shaped plate"/>
            <a href="#!">
                <span class="ipyplot-img-close"/>
            </a>
            <a href="#ipyplot-content-div-7j3Fk4SG4Jwxoafp6tRuiB-89y7xSULgpfyqd9QX8X9sf">
                <span class="ipyplot-img-expand"/>
            </a>
        </div>
    </div>

    <div class="ipyplot-placeholder-div-7j3Fk4SG4Jwxoafp6tRuiB">
        <div id="ipyplot-content-div-7j3Fk4SG4Jwxoafp6tRuiB-XDXTQZXDutnkP2UeJ89UQE" class="ipyplot-content-div-7j3Fk4SG4Jwxoafp6tRuiB">
            <h4 style="font-size: 12px; word-wrap: break-word;">2</h4>
            <img src="https://runhouse-tutorials.s3.amazonaws.com/pin_2.png" alt="image of a hot dog in a bun with matcha topping on a wooden surface"/>
            <a href="#!">
                <span class="ipyplot-img-close"/>
            </a>
            <a href="#ipyplot-content-div-7j3Fk4SG4Jwxoafp6tRuiB-XDXTQZXDutnkP2UeJ89UQE">
                <span class="ipyplot-img-expand"/>
            </a>
        </div>
    </div>

    <div class="ipyplot-placeholder-div-7j3Fk4SG4Jwxoafp6tRuiB">
        <div id="ipyplot-content-div-7j3Fk4SG4Jwxoafp6tRuiB-g5ofaCEvZ4UHffLp3H6evR" class="ipyplot-content-div-7j3Fk4SG4Jwxoafp6tRuiB">
            <h4 style="font-size: 12px; word-wrap: break-word;">3</h4>
            <img src="https://runhouse-tutorials.s3.amazonaws.com/pin_3.png" alt="image of a hot dog in a bun spread with matcha on a marble surface"/>
            <a href="#!">
                <span class="ipyplot-img-close"/>
            </a>
            <a href="#ipyplot-content-div-7j3Fk4SG4Jwxoafp6tRuiB-g5ofaCEvZ4UHffLp3H6evR">
                <span class="ipyplot-img-expand"/>
            </a>
        </div>
    </div>
    </div>


Runhouse Module
~~~~~~~~~~~~~~~

The second approach constructs a Runhouse Module that maintains the
model as a class variable. In notebook settings, we define this Module
in another file and import it here.

.. code:: ipython3

    # sd_model.py

   import runhouse as rh
   import torch
   from diffusers import StableDiffusionPipeline

   class SDModel(rh.Module):

       def __init__(self, model_id='stabilityai/stable-diffusion-2-base',
                          dtype=torch.float16, revision="fp16", device="cuda"):
           super().__init__()
           self.model_id, self.dtype, self.revision, self.device = model_id, dtype, revision, device

       @property
       def pipeline(self):
           if not hasattr(self, '_pipeline'):
               self._pipeline = StableDiffusionPipeline.from_pretrained(self.model_id, torch_dtype=self.dtype, revision=self.revision).to(self.device)
           return self._pipeline

       def predict(self, prompt, num_images=1, steps=100, guidance_scale=7.5):
           return self.pipeline(prompt, num_images_per_prompt=num_images,
                                num_inference_steps=steps, guidance_scale=guidance_scale).images

.. code:: ipython3

    from sd_model import SDModel

    model = SDModel()
    model_gpu = model.to(system=gpu)

    module_images = model_gpu.predict('my_prompt', num_images=4, steps=50)


FLAN-T5 Stable Diffusion
---------------------------

Here, we use FLAN-T5, a text-to-text generation model, to generate
prompts for us. We’ll send a FLAN-T5 inference function to our GPU, and
then pipe the outputs into our Stable Diffusion service.

.. code:: ipython3

    def causal_lm_generate(prompt, model_id='google/flan-t5-xl', **model_kwargs):
        import runhouse as rh
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        (tokenizer, model) = rh.get_pinned_object(model_id) or (None, None)

        # pin to memory if it is not in memory yet
        if model is None:
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            model = AutoModelForSeq2SeqLM.from_pretrained(model_id).to('cuda')
            rh.pin_to_memory(model_id, (tokenizer, model))
        inputs = tokenizer(prompt, return_tensors="pt").to('cuda')
        outputs = model.generate(**inputs, **model_kwargs)
        return tokenizer.batch_decode(outputs, skip_special_tokens=True)

.. code:: ipython3

    flan_t5_generate = rh.function(fn=causal_lm_generate).to(gpu, env=['./'])


.. parsed-literal::
    :class: code-output

    INFO | 2023-02-07 16:15:17,183 | Writing out function to /content/send_fn.py as functions serialized in notebooks are brittle. Please make sure the function does not rely on any local variables, including imports (which should be moved inside the function body).
    WARNING | 2023-02-07 16:15:17,186 | You should name Functions that are created in notebooks to avoid naming collisions between the modules that are created to hold their functions (i.e. "send_fn.py" errors.
    INFO | 2023-02-07 16:15:17,197 | Setting up Function on cluster.
    INFO | 2023-02-07 16:15:17,204 | Copying local package content to cluster <rh-a10x-aws>
    INFO | 2023-02-07 16:15:17,206 | Creating new ssh folder: content
    INFO | 2023-02-07 16:15:17,514 | Installing packages on cluster rh-a10x-aws: ['./']
    INFO | 2023-02-07 16:15:17,558 | Function setup complete.


.. code:: ipython3

    my_prompt = "A detailed oil painting of"
    sequences = flan_t5_generate(my_prompt, max_new_tokens=100, min_length=20, temperature=2.0, repetition_penalty=3.0,
                                  use_cache=False, do_sample=True, num_beams=3, num_return_sequences=4)

    full_seqs = [my_prompt + " " + seq for seq in sequences]
    for seq in full_seqs:
      print(seq)


.. parsed-literal::
    :class: code-output

    INFO | 2023-02-07 16:15:19,115 | Running causal_lm_generate via gRPC
    INFO | 2023-02-07 16:19:04,544 | Time to send message: 225.42 seconds
    A detailed oil painting of an ancient Greek vase in a medieval gallery with two candlesticks on and some sort of pedestal. It is signed at lower right "C.P." (probably the same signature used by the artist).
    A detailed oil painting of a rose in the sun. This beautiful flower is known as an achile, which means that it can only be seen by night-time predators like crows and robins. In this painting you'll notice all sorts of details on the leaves (that are not actually green), flowers, and butterflies. It's a small but lovely detail for someone who wants to have their art work professionally done.
    A detailed oil painting of horses and a cart on an idyllic country farm. Painted in 1880 by the painter, Auguste Bresson (1845-1925), for his own private collection; over time it became part of the Art Gallery of Victoria's permanent collection.
    A detailed oil painting of the ancient greek god, who is believed to have given his sons and daughters knowledge of magic tricks in order for them to excel in their art (the sorcerers were known as oracles)


.. code:: ipython3

    # We can directly access the function by the name we saved it by, even on a different environment or run
    saved_sd_generate = rh.function('sd_generate_pinned')
    paintings = saved_sd_generate(full_seqs, num_images=1, steps=50)

    ipyplot.plot_images(paintings)


.. parsed-literal::
    :class: code-output

    INFO | 2023-02-07 16:25:41,344 | Running sd_generate_pinned via gRPC
    INFO | 2023-02-07 16:26:20,268 | Time to send message: 38.92 seconds
    WARNING | 2023-02-07 16:26:20,280 | /usr/local/lib/python3.8/dist-packages/ipyplot/_utils.py:95: FutureWarning: The input object of type 'Image' is an array-like implementing one of the corresponding protocols (`__array__`, `__array_interface__` or `__array_struct__`); but not a sequence (or 0-D). In the future, this object will be coerced as if it was first converted using `np.array(obj)`. To retain the old behaviour, you have to either modify the type 'Image', or assign to an empty array created with `np.empty(correct_shape, dtype=object)`.
      return np.asarray(seq, dtype=type(seq[0]))




.. raw:: html


    <style>
        #ipyplot-html-viewer-toggle-5ZWRVjNXFdj5TVj7x4oWfn {
            position: absolute;
            top: -9999px;
            left: -9999px;
            visibility: hidden;
        }

        #ipyplot-html-viewer-label-5ZWRVjNXFdj5TVj7x4oWfn {
            position: relative;
            display: inline-block;
            cursor: pointer;
            color: blue;
            text-decoration: underline;
        }

        #ipyplot-html-viewer-textarea-5ZWRVjNXFdj5TVj7x4oWfn {
            background: lightgrey;
            width: 100%;
            height: 0px;
            display: none;
        }

        #ipyplot-html-viewer-toggle-5ZWRVjNXFdj5TVj7x4oWfn:checked ~ #ipyplot-html-viewer-textarea-5ZWRVjNXFdj5TVj7x4oWfn {
            height: 200px;
            display: block;
        }

        #ipyplot-html-viewer-toggle-5ZWRVjNXFdj5TVj7x4oWfn:checked + #ipyplot-html-viewer-label-5ZWRVjNXFdj5TVj7x4oWfn:after {
            content: "hide html";
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: white;
            cursor: pointer;
            color: blue;
            text-decoration: underline;
        }
    </style>
    <div>
        <input type="checkbox" id="ipyplot-html-viewer-toggle-5ZWRVjNXFdj5TVj7x4oWfn">
        <label id="ipyplot-html-viewer-label-5ZWRVjNXFdj5TVj7x4oWfn" for="ipyplot-html-viewer-toggle-5ZWRVjNXFdj5TVj7x4oWfn">show html</label>
        <textarea id="ipyplot-html-viewer-textarea-5ZWRVjNXFdj5TVj7x4oWfn" readonly>

        <style>
        #ipyplot-imgs-container-div-ctt8w42CeJB5xSoBbRX7Pi {
            width: 100%;
            height: 100%;
            margin: 0%;
            overflow: auto;
            position: relative;
            overflow-y: scroll;
        }

        div.ipyplot-placeholder-div-ctt8w42CeJB5xSoBbRX7Pi {
            width: 150px;
            display: inline-block;
            margin: 3px;
            position: relative;
        }

        div.ipyplot-content-div-ctt8w42CeJB5xSoBbRX7Pi {
            width: 150px;
            background: white;
            display: inline-block;
            vertical-align: top;
            text-align: center;
            position: relative;
            border: 2px solid #ddd;
            top: 0;
            left: 0;
        }

        div.ipyplot-content-div-ctt8w42CeJB5xSoBbRX7Pi span.ipyplot-img-close {
            display: none;
        }

        div.ipyplot-content-div-ctt8w42CeJB5xSoBbRX7Pi span {
            width: 100%;
            height: 100%;
            position: absolute;
            top: 0;
            left: 0;
        }

        div.ipyplot-content-div-ctt8w42CeJB5xSoBbRX7Pi img {
            width: 150px;
        }

        div.ipyplot-content-div-ctt8w42CeJB5xSoBbRX7Pi span.ipyplot-img-close:hover {
            cursor: zoom-out;
        }
        div.ipyplot-content-div-ctt8w42CeJB5xSoBbRX7Pi span.ipyplot-img-expand:hover {
            cursor: zoom-in;
        }

        div[id^=ipyplot-content-div-ctt8w42CeJB5xSoBbRX7Pi]:target {
            transform: scale(2.5);
            transform-origin: left top;
            z-index: 5000;
            top: 0;
            left: 0;
            position: absolute;
        }

        div[id^=ipyplot-content-div-ctt8w42CeJB5xSoBbRX7Pi]:target span.ipyplot-img-close {
            display: block;
        }

        div[id^=ipyplot-content-div-ctt8w42CeJB5xSoBbRX7Pi]:target span.ipyplot-img-expand {
            display: none;
        }
        </style>
    <div id="ipyplot-imgs-container-div-ctt8w42CeJB5xSoBbRX7Pi">
    <div class="ipyplot-placeholder-div-ctt8w42CeJB5xSoBbRX7Pi">
        <div id="ipyplot-content-div-ctt8w42CeJB5xSoBbRX7Pi-6vBxQ9aAKu9e5dBydz3wnd" class="ipyplot-content-div-ctt8w42CeJB5xSoBbRX7Pi">
            <h4 style="font-size: 12px; word-wrap: break-word;">0</h4>
            <img src="https://runhouse-tutorials.s3.amazonaws.com/flan_0.png" alt="generated image of a detailed oil painting of an ancient Greek vase"/>
            <a href="#!">
                <span class="ipyplot-img-close"/>
            </a>
            <a href="#ipyplot-content-div-ctt8w42CeJB5xSoBbRX7Pi-6vBxQ9aAKu9e5dBydz3wnd">
                <span class="ipyplot-img-expand"/>
            </a>
        </div>
    </div>

    <div class="ipyplot-placeholder-div-ctt8w42CeJB5xSoBbRX7Pi">
        <div id="ipyplot-content-div-ctt8w42CeJB5xSoBbRX7Pi-iNHddryXJPQBpaDXBjqMxN" class="ipyplot-content-div-ctt8w42CeJB5xSoBbRX7Pi">
            <h4 style="font-size: 12px; word-wrap: break-word;">1</h4>
            <img src="https://runhouse-tutorials.s3.amazonaws.com/flan_1.png" alt="generated image of a detailed oil painting of a rose in the sun"/>
            <a href="#!">
                <span class="ipyplot-img-close"/>
            </a>
            <a href="#ipyplot-content-div-ctt8w42CeJB5xSoBbRX7Pi-iNHddryXJPQBpaDXBjqMxN">
                <span class="ipyplot-img-expand"/>
            </a>
        </div>
    </div>

    <div class="ipyplot-placeholder-div-ctt8w42CeJB5xSoBbRX7Pi">
        <div id="ipyplot-content-div-ctt8w42CeJB5xSoBbRX7Pi-efUNKGLZXEg2WPbtyMoNLr" class="ipyplot-content-div-ctt8w42CeJB5xSoBbRX7Pi">
            <h4 style="font-size: 12px; word-wrap: break-word;">2</h4>
            <img src="https://runhouse-tutorials.s3.amazonaws.com/flan_2.png" alt="generated image of a detailed oil painting of horses and a cart on an idyllic country farm"/>
            <a href="#!">
                <span class="ipyplot-img-close"/>
            </a>
            <a href="#ipyplot-content-div-ctt8w42CeJB5xSoBbRX7Pi-efUNKGLZXEg2WPbtyMoNLr">
                <span class="ipyplot-img-expand"/>
            </a>
        </div>
    </div>

    <div class="ipyplot-placeholder-div-ctt8w42CeJB5xSoBbRX7Pi">
        <div id="ipyplot-content-div-ctt8w42CeJB5xSoBbRX7Pi-MN7beZHJBF6X42xSNzZ5Gb" class="ipyplot-content-div-ctt8w42CeJB5xSoBbRX7Pi">
            <h4 style="font-size: 12px; word-wrap: break-word;">3</h4>
            <img src="https://runhouse-tutorials.s3.amazonaws.com/flan_3.png" alt="generated image of a detailed oil painting of the ancient greek god"/>
            <a href="#!">
                <span class="ipyplot-img-close"/>
            </a>
            <a href="#ipyplot-content-div-ctt8w42CeJB5xSoBbRX7Pi-MN7beZHJBF6X42xSNzZ5Gb">
                <span class="ipyplot-img-expand"/>
            </a>
        </div>
    </div>
    </div>
        </textarea>
    </div>




.. raw:: html


        <style>
        #ipyplot-imgs-container-div-ctt8w42CeJB5xSoBbRX7Pi {
            width: 100%;
            height: 100%;
            margin: 0%;
            overflow: auto;
            position: relative;
            overflow-y: scroll;
        }

        div.ipyplot-placeholder-div-ctt8w42CeJB5xSoBbRX7Pi {
            width: 150px;
            display: inline-block;
            margin: 3px;
            position: relative;
        }

        div.ipyplot-content-div-ctt8w42CeJB5xSoBbRX7Pi {
            width: 150px;
            background: white;
            display: inline-block;
            vertical-align: top;
            text-align: center;
            position: relative;
            border: 2px solid #ddd;
            top: 0;
            left: 0;
        }

        div.ipyplot-content-div-ctt8w42CeJB5xSoBbRX7Pi span.ipyplot-img-close {
            display: none;
        }

        div.ipyplot-content-div-ctt8w42CeJB5xSoBbRX7Pi span {
            width: 100%;
            height: 100%;
            position: absolute;
            top: 0;
            left: 0;
        }

        div.ipyplot-content-div-ctt8w42CeJB5xSoBbRX7Pi img {
            width: 150px;
        }

        div.ipyplot-content-div-ctt8w42CeJB5xSoBbRX7Pi span.ipyplot-img-close:hover {
            cursor: zoom-out;
        }
        div.ipyplot-content-div-ctt8w42CeJB5xSoBbRX7Pi span.ipyplot-img-expand:hover {
            cursor: zoom-in;
        }

        div[id^=ipyplot-content-div-ctt8w42CeJB5xSoBbRX7Pi]:target {
            transform: scale(2.5);
            transform-origin: left top;
            z-index: 5000;
            top: 0;
            left: 0;
            position: absolute;
        }

        div[id^=ipyplot-content-div-ctt8w42CeJB5xSoBbRX7Pi]:target span.ipyplot-img-close {
            display: block;
        }

        div[id^=ipyplot-content-div-ctt8w42CeJB5xSoBbRX7Pi]:target span.ipyplot-img-expand {
            display: none;
        }
        </style>
    <div id="ipyplot-imgs-container-div-ctt8w42CeJB5xSoBbRX7Pi">
    <div class="ipyplot-placeholder-div-ctt8w42CeJB5xSoBbRX7Pi">
        <div id="ipyplot-content-div-ctt8w42CeJB5xSoBbRX7Pi-6vBxQ9aAKu9e5dBydz3wnd" class="ipyplot-content-div-ctt8w42CeJB5xSoBbRX7Pi">
            <h4 style="font-size: 12px; word-wrap: break-word;">0</h4>
            <img src="https://runhouse-tutorials.s3.amazonaws.com/flan_0.png" alt="generated image of a detailed oil painting of an ancient Greek vase"/>
            <a href="#!">
                <span class="ipyplot-img-close"/>
            </a>
            <a href="#ipyplot-content-div-ctt8w42CeJB5xSoBbRX7Pi-6vBxQ9aAKu9e5dBydz3wnd">
                <span class="ipyplot-img-expand"/>
            </a>
        </div>
    </div>

    <div class="ipyplot-placeholder-div-ctt8w42CeJB5xSoBbRX7Pi">
        <div id="ipyplot-content-div-ctt8w42CeJB5xSoBbRX7Pi-iNHddryXJPQBpaDXBjqMxN" class="ipyplot-content-div-ctt8w42CeJB5xSoBbRX7Pi">
            <h4 style="font-size: 12px; word-wrap: break-word;">1</h4>
            <img src="https://runhouse-tutorials.s3.amazonaws.com/flan_1.png" alt="generated image of a detailed oil painting of a rose in the sun"/>
            <a href="#!">
                <span class="ipyplot-img-close"/>
            </a>
            <a href="#ipyplot-content-div-ctt8w42CeJB5xSoBbRX7Pi-iNHddryXJPQBpaDXBjqMxN">
                <span class="ipyplot-img-expand"/>
            </a>
        </div>
    </div>

    <div class="ipyplot-placeholder-div-ctt8w42CeJB5xSoBbRX7Pi">
        <div id="ipyplot-content-div-ctt8w42CeJB5xSoBbRX7Pi-efUNKGLZXEg2WPbtyMoNLr" class="ipyplot-content-div-ctt8w42CeJB5xSoBbRX7Pi">
            <h4 style="font-size: 12px; word-wrap: break-word;">2</h4>
            <img src="https://runhouse-tutorials.s3.amazonaws.com/flan_2.png" alt="generated image of a detailed oil painting of horses and a cart on an idyllic country farm"/>
            <a href="#!">
                <span class="ipyplot-img-close"/>
            </a>
            <a href="#ipyplot-content-div-ctt8w42CeJB5xSoBbRX7Pi-efUNKGLZXEg2WPbtyMoNLr">
                <span class="ipyplot-img-expand"/>
            </a>
        </div>
    </div>

    <div class="ipyplot-placeholder-div-ctt8w42CeJB5xSoBbRX7Pi">
        <div id="ipyplot-content-div-ctt8w42CeJB5xSoBbRX7Pi-MN7beZHJBF6X42xSNzZ5Gb" class="ipyplot-content-div-ctt8w42CeJB5xSoBbRX7Pi">
            <h4 style="font-size: 12px; word-wrap: break-word;">3</h4>
            <img src="https://runhouse-tutorials.s3.amazonaws.com/flan_3.png" alt="generated image of a detailed oil painting of the ancient greek god"/>
            <a href="#!">
                <span class="ipyplot-img-close"/>
            </a>
            <a href="#ipyplot-content-div-ctt8w42CeJB5xSoBbRX7Pi-MN7beZHJBF6X42xSNzZ5Gb">
                <span class="ipyplot-img-expand"/>
            </a>
        </div>
    </div>
    </div>
