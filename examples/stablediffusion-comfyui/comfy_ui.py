import runhouse as rh

if __name__ == "__main__":
    img = (
        rh.Image(name="sdxl_inference")
        .pip_install(
            [
                "diffusers",
                "torch",
                "torchvision",
                "torchaudio",
                "transformers",
                "accelerate",
                "comfy-cli",
            ]
        )
        .sync_secrets(["huggingface"])
    )

    cluster = rh.compute(
        name="rh-g5",
        instance_type="g5.8xlarge",
        provider="aws",
        image=img,
    ).up_if_not()

    cluster.run_bash_over_ssh(
        [
            "git clone https://github.com/comfyanonymous/ComfyUI.git && cd ComfyUI && pip install -r requirements.txt"
        ]
    )
    cluster.ssh_tunnel(8188, 8188)  # Now accessible locally on port 8188
    cluster.run_bash_over_ssh(
        [
            "cd ComfyUI && comfy --skip-prompt tracking disable && comfy model download --url https://huggingface.co/Comfy-Org/stable-diffusion-v1-5-archive/resolve/main/v1-5-pruned-emaonly-fp16.safetensors?download=true --relative-path models/checkpoints --filename v1-5-pruned-emaonly-fp16.safetensors"
        ]
    )
    cluster.run_bash_over_ssh(["cd ComfyUI && python main.py"])
