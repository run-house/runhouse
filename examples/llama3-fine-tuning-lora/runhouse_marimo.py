import marimo

__generated_with = "0.10.12"
app = marimo.App(width="medium")


@app.cell
def _():
    # Launch a Llama 3 fine-tuning task with marimo and Runhouse
    # In this example, you will launch a remote cluster, write your training in a marimo cell, and then send your training code to that remote compute to run. This way, you can access unlimited power and scaling potential while still having a great dev experience.
    return


app._unparsable_cell(
    r"""
    # Setup steps, simply importing marimo and Runhouse libraries and setting your HuggingFace Token which is needed to download the libraries. We assume by opening this, you already have marimo installed; Runhouse is installed with the additional `aws` though you could do `azure` or `gcp` instead
    !pip install \"runhouse[aws]\"
    import marimo as mo
    import runhouse as rh

    import os
    os.environ[\"HF_TOKEN\"] = \"My Hugging Face Token\" # Used to download Llama3 weights from HF - make sure you sign the consent form
    """,
    name="_",
)


@app.cell
def _(rh):
    # Launch a remote cluster with a GPU - here we use an L4 from AWS. We assume you have already set up with your cloud provider CLI (e.g. `aws configure` or `gcloud init`) and your account is authorized to launch compute.
    img = (
        rh.Image(name="llama3finetuning")
        .install_packages(
            [
                "torch",
                "tensorboard",
                "transformers",
                "bitsandbytes",
                "peft",
                "trl",
                "accelerate",
                "scipy",
                "marimo",
            ]
        )
        .sync_secrets(["huggingface"])
    )

    cluster = rh.cluster(
        name="rh-L4", instance_type="L4:1", provider="aws", image=img, autostop_mins=120
    ).up_if_not()
    return cluster, img


@app.cell(disabled=True)
def finetune():
    import gc
    from pathlib import Path

    import torch
    from accelerate import Accelerator, PartialState
    from datasets import load_dataset
    from peft import AutoPeftModelForCausalLM, LoraConfig

    # Define the fine tuning in this cell that we will dispatch and run in the remote cluster we launched. This cell is named "finetuner" so we can dispatch it by function name in the next step

    # Dispatching and executing will happen in the next cell - but by making edits here and rerunning the next cell, your iteration loops should feel local-like (dispatch takes <2s to remote)
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        pipeline,
    )
    from trl import SFTConfig, SFTTrainer

    DEFAULT_MAX_LENGTH = 200

    DATASET_NAME = "mlabonne/guanaco-llama2-1k"
    BASE_MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"
    FINE_TUNED_MODEL_NAME = "llama-3-3b-enhanced"

    def load_base_model(base_model_name):
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=False,
        )

        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name, quantization_config=quant_config, device_map={"": 0}
        )

        base_model.config.use_cache = False
        base_model.config.pretraining_tp = 1
        return base_model

    def load_tokenizer(base_model_name):
        tokenizer = AutoTokenizer.from_pretrained(
            base_model_name, trust_remote_code=True
        )
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        return tokenizer

    def load_pipeline(fine_tuned_model, tokenizer, max_length):
        return pipeline(
            task="text-generation",
            model=fine_tuned_model,
            tokenizer=tokenizer,
            max_length=max_length,
        )

    def load_dataset_data(dataset_name):
        return load_dataset(dataset_name, split="train")

    def training_params():
        return SFTConfig(
            output_dir="./results_modified",
            num_train_epochs=1,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=1,
            optim="paged_adamw_32bit",
            save_steps=25,
            logging_steps=25,
            learning_rate=2e-4,
            weight_decay=0.001,
            fp16=False,
            bf16=False,
            max_grad_norm=0.3,
            max_steps=-1,
            warmup_ratio=0.03,
            dataset_text_field="text",
            group_by_length=True,
            lr_scheduler_type="constant",
            report_to="tensorboard",
        )

    def sft_trainer(
        base_model, training_data, peft_parameters, tokenizer, train_params
    ):
        return SFTTrainer(
            model=base_model,
            train_dataset=training_data,
            peft_config=peft_parameters,
            tokenizer=tokenizer,
            args=train_params,
        )

    def tune(base_model_name, dataset_name, fine_tuned_model_name):
        gc.collect()
        torch.cuda.empty_cache()
        gc.collect()

        training_data = load_dataset_data(dataset_name)
        tokenizer = load_tokenizer(base_model_name)
        base_model = load_base_model(base_model_name)

        peft_parameters = LoraConfig(
            lora_alpha=16, lora_dropout=0.1, r=8, bias="none", task_type="CAUSAL_LM"
        )

        train_params = training_params()
        trainer = sft_trainer(
            base_model, training_data, peft_parameters, tokenizer, train_params
        )

        trainer.train()

        trainer.model.save_pretrained(fine_tuned_model_name)
        trainer.tokenizer.save_pretrained(fine_tuned_model_name)
        print("Saved model weights and tokenizer on the cluster.")

    def load_fine_tuned_model(fine_tuned_model_name):
        if not Path(f"~/{fine_tuned_model_name}").expanduser().exists():
            raise FileNotFoundError(
                "No fine-tuned model found on the cluster. Call the `tune` method to run the fine-tuning."
            )

        fine_tuned_model = AutoPeftModelForCausalLM.from_pretrained(
            fine_tuned_model_name,
            device_map={"": "cuda:0"},
            torch_dtype=torch.bfloat16,
        )
        return fine_tuned_model.merge_and_unload()

    def generate(query, fine_tuned_model_name, max_length=DEFAULT_MAX_LENGTH):
        fine_tuned_model = load_fine_tuned_model(fine_tuned_model_name)
        tokenizer = load_tokenizer(BASE_MODEL_NAME)
        gen_pipeline = load_pipeline(fine_tuned_model, tokenizer, max_length)

        output = gen_pipeline(f"<s>[INST] {query} [/INST]")
        return output[0]["generated_text"]

    tune(BASE_MODEL_NAME, DATASET_NAME, FINE_TUNED_MODEL_NAME)
    print(generate("What is the capital of France?", FINE_TUNED_MODEL_NAME))
    return (
        Accelerator,
        AutoModelForCausalLM,
        AutoPeftModelForCausalLM,
        AutoTokenizer,
        BASE_MODEL_NAME,
        BitsAndBytesConfig,
        DATASET_NAME,
        DEFAULT_MAX_LENGTH,
        FINE_TUNED_MODEL_NAME,
        LoraConfig,
        PartialState,
        Path,
        SFTConfig,
        SFTTrainer,
        gc,
        generate,
        load_base_model,
        load_dataset,
        load_dataset_data,
        load_fine_tuned_model,
        load_pipeline,
        load_tokenizer,
        pipeline,
        sft_trainer,
        torch,
        training_params,
        tune,
    )


@app.cell
def _(mo):
    # Some parameters that are available for use in training
    train_params = mo.md(
        """{epochs} \n
    {per_device_train_batch_size} \n
    {gradient_accumulation_steps} \n
    {optimizer} \n
    {learning_rate} \n
    {output_dir} \n
    {dataset_text_field}
    """
    ).batch(
        epochs=mo.ui.number(start=1, stop=100, step=1, label="Epochs"),
        per_device_train_batch_size=mo.ui.number(
            start=1, stop=100, step=1, label="Batch Size"
        ),
        gradient_accumulation_steps=mo.ui.slider(
            start=1, stop=10, step=1, label="Gradient Accumulation Steps"
        ),
        optimizer=mo.ui.dropdown(
            options=["paged_adamw_32bit", "paged_adamw_8bit"],
            value="paged_adamw_32bit",
            label="Optimizer",
        ),
        learning_rate=mo.ui.number(
            start=0, stop=1, value=2e-4, label="Learn Rate", step=0.0001
        ),
        output_dir=mo.ui.text(value="./results_modified", label="Output Directory"),
        dataset_text_field=mo.ui.text(value="text", label="Dataset Text Field"),
    )

    train_params
    return (train_params,)


@app.cell
def _(cluster, rh):
    # from llama3_fine_tuning import FineTuner
    from runhouse_marimo import finetune

    fine_tuner_remote = rh.function(finetune).to(cluster, name="ft_model")
    fine_tuner_remote()
    return fine_tuner_remote, finetune


if __name__ == "__main__":
    app.run()
