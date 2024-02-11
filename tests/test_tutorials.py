import runhouse as rh


def sd_generate(
    prompt,
    num_images=1,
    steps=100,
    guidance_scale=7.5,
    model_id="stabilityai/stable-diffusion-2-base",
):
    import torch
    from diffusers import StableDiffusionPipeline

    pipe = StableDiffusionPipeline.from_pretrained(
        model_id, torch_dtype=torch.float16, revision="fp16"
    ).to("cuda")
    return pipe(
        [prompt] * num_images, num_inference_steps=steps, guidance_scale=guidance_scale
    ).images


def test_sd_generate(a10g_gpu_cluster):
    generate_gpu = rh.function(fn=sd_generate).to(
        a10g_gpu_cluster, reqs=["pytest", "diffusers", "torch", "transformers"]
    )

    images = generate_gpu(
        prompt="A hot dog made of matcha powder.", num_images=4, steps=50
    )
    assert images
