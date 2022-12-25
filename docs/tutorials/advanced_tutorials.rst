Advanced Tutorials
====================================

`Check out a few of our more advanced tutorials`


DALL-E Mini to Stable Diffusion
------------------------------------

`In this tutorial, we will create seed images with DALL-E Mini and then use Stable Diffusion img2img to refine them.`


We'll start by defining the initial text prompt and our DALL-E function to generate the seed images:

.. code-block:: python

    rh_prompt = 'A digital illustration of a woman running on the roof of a house.'

    def dm_generate(prompt, num_images_sqrt=1, supercondition_factor=32,
                    is_mega=True, seed=50, top_k=64):
        from min_dalle import MinDalle
        torch.cuda.empty_cache()
        torch.no_grad()
        dalle = MinDalle(device='cuda', is_mega=is_mega, is_reusable=False, dtype=torch.float16)
        images = dalle.generate_images(prompt, seed=seed, grid_size=num_images_sqrt,
                                       temperature=1, top_k=top_k, supercondition_factor=supercondition_factor)
        images = images.to(torch.uint8).to('cpu').numpy()
        return [Image.fromarray(images[i]) for i in range(num_images_sqrt**2)]


Next, we'll create our Stable Diffusion generate function:

.. code-block:: python

    def sd_img2img_generate(prompt, base_images, num_images=1,
                            steps=100, strength=0.75, guidance_scale=7.5,
                            model_id="runwayml/stable-diffusion-v1-5"):
        from diffusers import StableDiffusionImg2ImgPipeline
        torch.cuda.empty_cache()
        torch.no_grad()
        sd_pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_id)
        sd_pipe = sd_pipe.to('cuda')
        ret = []
        for image in base_images:
            ret = ret + sd_pipe([prompt] * num_images, init_image=image.resize((512, 512)),
                                num_inference_steps=steps, strength=strength,
                                guidance_scale=guidance_scale).images
        return ret


Now that we have our functions defined we can create our Dalle-E Mini send:

.. code-block:: python

    # First try just DALL-E Mini. It runs best on an A100, but AWS doesn't offer single A100s (only clusters of 8),
    # so let's run it on gcp.
    dm_generate_gpu = rh.send(fn=dm_generate,
                              hardware='^rh-v100',
                              reqs=['requirements.txt', 'min-dalle'])

    rh_logo_dm_images = dm_generate_gpu(rh_prompt, seed=random.randint(0, 1000),
                                        num_images_sqrt=2, supercondition_factor=256)
    [image.show() for image in rh_logo_dm_images]


Finally, we can create a separate cluster and send for our Stable Diffusion code:

.. code-block:: python

    # Now let's try feeding it into StableDiffusionImg2Img.
    # We could put this on a the A100, but it might lead to OOM error,
    # so let's put it on a V100 on AWS.
    aws_gpu = rh.cluster(name='v100',
                         instance_type='V100:1',
                         provider='aws',
                         use_spot=True,
                         autostop_mins=60)
    sd_img2img_generate_gpu = rh.send(fn=sd_img2img_generate,
                                      hardware=aws_gpu,
                                      reqs=['requirements.txt', 'diffusers'],
                                      load_secrets=True)

Now all that's left is to build our "pipeline" and run it:

.. code-block:: python

    rh_logo_dm2sd_images = sd_img2img_generate_gpu(rh_prompt, rh_logo_dm_images, strength=.75,
                                                   guidance_scale=7.5, steps=25)
    [image.show() for image in rh_logo_dm2sd_images]

    # Now let's do a tester passing an existing runhouse logo image to SDImg2Img
    rh_base_image = Image.open('rh_logo.png').convert("RGB").resize((512, 512))
    rh_logo_sd_images = sd_img2img_generate_gpu(rh_prompt, [rh_base_image],
                                                strength=.5, guidance_scale=5,
                                                num_images=4, steps=100)
    [image.show() for image in rh_logo_sd_images]

.. note::
    This tutorial also shows how easy it is to deploy sends to different providers within the same pipeline.
    Here we ran our DALL-E Mini send on GCP, and our Stable Diffusion send on AWS!

BERT Fine Tuning
------------------------------------

In this example we will create a BERT fine tuning send. The data will be streamed in
via a table object, and the resulting model will be saved to a blob object.

For more information on the Runhouse data primitives, see our :ref:`intro tutorial <Data Layer>`.

.. code-block:: python

    from transformers import AutoModelForSequenceClassification, get_scheduler
    from accelerate import Accelerator
    import torch
    from tqdm.auto import tqdm  # progress bar

    def fine_tune_bert(preprocessed_data_name, num_epochs, model_out_name):
        # Hugging Face Accelerate automatically detects our distributed hardware and sets up
        # Distributed Data Parallelism across our GPUs.
        # https://huggingface.co/docs/transformers/accelerate
        accelerator = Accelerator()
        model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased",
                                                                   num_labels=5)
        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

        # Stream the data from the URI in batches rather than loading in-memory
        preprocessed_data = rh.Table(name=preprocessed_data_name)
        train_dataloader = preprocessed_data['train'].stream(shuffle=True, batch_size=8)
        eval_dataloader = preprocessed_data['test'].stream(batch_size=8)

        train_dataloader, eval_dataloader, model, optimizer = accelerator.prepare(
            train_dataloader, eval_dataloader, model, optimizer
        )

        num_training_steps = epochs * len(train_dataloader)
        lr_scheduler = get_scheduler(name="linear",
                                     optimizer=optimizer,
                                     num_warmup_steps=0,
                                     num_training_steps=num_training_steps)
        progress_bar = tqdm(range(num_training_steps))

        model.train()
        for epoch in range(num_epochs):
            for batch in train_dataloader:
                outputs = model(**batch)
                loss = outputs.loss
                accelerator.backward(loss)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)

            metric = load_metric("accuracy")
            model.eval()
            for batch in eval_dataloader:
                with torch.no_grad():
                    outputs = model(**batch)

                logits = outputs.logits
                predictions = torch.argmax(logits, dim=-1)
                metric.add_batch(predictions=predictions, references=batch["labels"])

            metric.compute()
            print(f'Epoch {epoch} accuracy: {metric}')

        # Save the model to cloud blob storage for reuse later or sharing
        return rh.Blob(data=pickle.dumps(model), name=model_out_name)


Now initialize your send as follows:

.. code-block:: python

    # Send it to gpu hardware
    bert_ft = rh.Send(fn=fine_tune_bert,
                      hardware='my_8_gpus',
                      name='BERT_finetune_8gpu'

.. note::
    The hardware we specified here is not a Runhouse builtin hardware type. Runhouse supports custom hardware
    specifications. See our guide on :ref:`clusters <cluster>` for more information on defining hardware types.

Finally, we can run our send. Notice how are function returns a Runhouse blob object. This means that once we run
our send, the results will be saved directly to blob storage.


.. code-block:: python

    trained_model = bert_ft(preprocessed_data_name='yelp_bert_preprocessed',
                            num_epochs=3,
                            model_out_name='yelp_fine_tuned_bert')


.. tip::
    Runhouse provides convenient defaults for blobs - by default they are saved as parquet in s3. However, you
    can easily override these defaults by updating the config for the resource.
    See our guide on :ref:`blobs <blob>` for more information.
