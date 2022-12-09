From Zero to Hero in 10 minutes 🚀
====================================

Check out a few of our tutorials to help you get going quickly


Stable Diffusion
------------------------------------
`Let's review how we can use Runhouse to easily generate images with Stable Diffusion.`

.. note::
    If you'd prefer to view this in a notebook environment, you can find it `here <https://github.com/run-house/tutorials/>`_.

For the first step, we'll define our generate function, which does two things:

- Initializes a pipeline to generate an image.
- Pipes the prompt and generates the desired number of images.

.. code-block:: python

    import runhouse as rh
    from diffusers import StableDiffusionPipeline
    import torch

    def sd_generate(prompt,
                    num_images=1,
                    steps=100,
                    guidance_scale=7.5,
                    model_id="runwayml/stable-diffusion-v1-5"):
        pipe = StableDiffusionPipeline.from_pretrained(model_id,
                                                       torch_dtype=torch.float16,
                                                       revision="fp16").to("cuda")
        return pipe([prompt] * num_images,
                    num_inference_steps=steps,
                    guidance_scale=guidance_scale).images

Next we create our **cluster**:

.. code-block:: python

    gpu = rh.cluster(name='rh-v100',
                     provider='gcp',
                     instance_type='V100:1',
                     use_spot=True)

- :code:`name`: Naming our cluster so we can re-use it later.
- :code:`provider`: Here we are using gcp, but you can also choose to use aws or azure.
- :code:`instance_type`: Cluster instance type. Here we are using a single :code:`V100` GPU.
- :code:`use_spot`: Use spot instance.

.. tip::
    In this example we chose to use a :code:`v100`, but Runhouse provides various convenient builtin hardware types.
    You can see all the supported options in the :ref:`send`.


Finally, we create of our **send**:

.. code-block:: python

    generate_gpu = rh.send(fn=sd_generate,
                           hardware=gpu,
                           reqs=['./', 'diffusers'],
                           load_secrets=True)

- :code:`fn`: Function we defined above.
- :code:`hardware`: Cluster object we created above.
- :code:`reqs`: Here we are using the existing requirements defined in our :code:`requirements.txt` file, add adding :code:`diffusers`.
- :code:`load_secrets`: Runhouse will upload your locally configured secrets to store securely in Vault (More on that in :ref:`Secrets Management`).

Now that we have our function, cluster, and send defined we can run it with some custom text.

.. code-block:: python

    rh_prompt = 'A digital illustration of a woman running on the roof of a house.'
    images = generate_gpu(rh_prompt, num_images=4, steps=50)

>>> print([image.show() for image in images])


Tokenizer
------------------------------------
`Let's review how we can use Runhouse to easily tokenize a Huggingface dataset.`

.. note::
    If you'd prefer to view this in a notebook environment, you can find it `here <https://github.com/run-house/tutorials/>`_.


For the first step, we'll load our tokenizer and define two functions:

- :code:`tokenize_function`: simple tokenizer function, which receives example data as input.
- :code:`tokenize_dataset`: tokenizes the entire dataset, parallelized across the number of available CPUs.

.. code-block:: python

    import os
    import runhouse as rh
    from transformers import AutoTokenizer  # Hugging face's model library
    from datasets import load_dataset, load_metric  # Hugging face's data library

    tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')

    def tokenize_function(examples):
        return tokenizer(examples, padding="max_length", truncation=True)

    def tokenize_dataset():
        hf_dataset = load_dataset("yelp_review_full", split=['test'])
        # Tokenizing the dataset serially can take a while, let's do it on across multiple CPUs
        return hf_dataset[0].map(tokenize_function,
                                 input_columns=['text'],
                                 num_proc=os.cpu_count(),
                                 batched=True)

Now that we have our functions defined we can create our send.

.. code-block:: python

    preproc = rh.Send(fn=tokenize_dataset,
                      name="tokenizer",
                      hardware="^rh-32-cpu")

.. note::
    We're using Runhouse's built-in 32-cpu hardware type here, and the container image was automatically
    inferred from the :code:`requirements.txt` that sits in your local working directory.
    Alternatively, you can provide a custom image, packages, or hardware.