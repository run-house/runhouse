# ## BERT Training Example
# This example demonstrates how to use Runhouse to further train a BERT model on a remote cluster with multiple GPUs.
# Our goal here is to add significant contextual knowledge of the medical domain with a PubMed dataset used to train
# MedBERT - https://huggingface.co/datasets/MedRAG/pubmed. We use ModernBert as a base model due to ModernBERT's ability to
# use long context windows (8192 tokens) as opposed to other BERTs which have much smaller context windows.
# The training will be distributed across multiple nodes and GPUs using PyTorch's distributed training capabilities.
import os
from pathlib import Path

import runhouse as rh
import s3fs
import torch
import torch.distributed as dist
from datasets import Dataset, DatasetDict
from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

# Helper function used to download the preprocessed data we generated from preprocess.py
def download_data(
    local_dir="./pubmed_processed",
    s3_path="s3://rh-demo-external/pubmed_processed/sample_10000",
    splits=["train", "eval", "test"],
    force_reload=False,
):

    local_dir = Path(local_dir)
    fs = s3fs.S3FileSystem()

    # Create local directory if it doesn't exist
    local_dir.mkdir(parents=True, exist_ok=True)

    for split in splits:
        local_path = local_dir / f"{split}.parquet"
        s3_path_download = f"{s3_path}/{split}"

        # Check if file exists locally
        if not local_path.exists() or force_reload:
            print(f"{split} split not found locally. Downloading from S3...")
            # Download from S3 to local
            fs.get(s3_path_download, str(local_path))
        else:
            print(f"Found {split} split locally.")


# Callback to save the model after each epoch
class SaveModelCallback(TrainerCallback):
    def __init__(self, trainer, output_dir):
        self.trainer = trainer
        self.output_dir = output_dir

    def on_epoch_end(self, args, state, control, **kwargs):
        print(f"Epoch {state.epoch} finished. Saving model...")
        self.trainer.save_model(self.output_dir)


# ## Trainer Class
# This class encapsulates the training process for a BERT model. This is standard
# training code that leverages Transformers/Accelerate and PyTorch's distributed training capabilities.
class BertTrainer:
    def __init__(
        self,
        model_name="answerdotai/ModernBERT-base",
        batch_size=32,
        max_length=128,
        learning_rate=2e-5,
        num_epochs=3,
    ):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.processed_dataset = None
        self.train_args = None
        self.trainer = None
        self.rank = None
        self.device = None
        self.local_rank = None

    def load_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

    def load_model(self):
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.device = torch.device(
            f"cuda:{self.local_rank}" if torch.cuda.is_available() else "cpu"
        )
        print(self.device)
        self.model = AutoModelForMaskedLM.from_pretrained(
            self.model_name, torch_dtype=torch.bfloat16
        )
        self.model.to(self.device)
        

    def load_data(
        self,
        local_dir="./pubmed_processed",
        s3_path="s3://rh-demo-external/pubmed_processed/sample_10000",
        splits=["train", "eval", "test"],
        force_reload=False,
    ):

        local_dir = Path(local_dir)
        datasets = {}
        for split in splits:
            local_path = local_dir / f"{split}.parquet"
            download_data(
                local_dir=local_dir,
                s3_path=s3_path,
                splits=splits,
                force_reload=force_reload,
            )
            # if not local_path.exists():
            #     raise Exception("Data not found locally. Download it first")

            datasets[split] = Dataset.from_parquet(str(local_path))

        self.processed_dataset = DatasetDict(datasets)
        print(self.processed_dataset)

    def train_helper(self):
        import gc

        gc.collect()
        torch.cuda.empty_cache()
        gc.collect()

        os.environ[
            "OMP_NUM_THREADS"
        ] = "1"  # Runhouse might separately have set other values of OMP_NUM_THREADS

        torch.distributed.init_process_group(backend="nccl")

        local_rank = str(
            dist.get_rank() % torch.cuda.device_count()
        )  # Set LOCAL_RANK by taking modulo of Rank and Device Count
        os.environ["LOCAL_RANK"] = local_rank

    def train(self, batch_size=32, max_length=128, learning_rate=2e-5, num_epochs=3):
        self.train_helper()

        if self.processed_dataset is None:
            self.load_data()

        if self.model is None:
            print("Loading model")
            self.load_model()

        if self.tokenizer is None:
            print("Loading tokenizer")
            self.load_tokenizer()

        self.train_args = TrainingArguments(
            output_dir="./bert_output",
            overwrite_output_dir=True,
            evaluation_strategy="epoch",
            optim="adamw_torch",
            save_strategy="epoch",
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            num_train_epochs=num_epochs,
            logging_dir="./logs",
            learning_rate=learning_rate,
            save_steps=1000,
            warmup_steps=100,
            report_to="tensorboard",
            ddp_backend="nccl",
            use_cpu=False,
        )
        print("Training Args Set")

        save_model_callback = SaveModelCallback(self, output_dir="./bert_output")

        print(
            f"Local Rank {self.local_rank} Model device {self.model.device}"
        )  # Check current state

        self.trainer = Trainer(
            model=self.model,
            args=self.train_args,
            train_dataset=self.processed_dataset["train"],
            eval_dataset=self.processed_dataset["eval"],
            callbacks=[save_model_callback],
        )
        self.trainer.train()

    def save_model(self, output_dir):
        if self.model is None:
            raise ValueError("No model to save - train first")
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)

        fs = s3fs.S3FileSystem()
        fs.put(
            output_dir,
            "s3://rh-demo-external/pubmed_processed/checkpoints/",
            recursive=True,
        )


# ## Launch Compute and Run Training
# Now we use Runhouse to launch a cluster with multiple nodes and GPUs and run the training on the remote machine.
# First, we define the RH image that will be used to install the necessary packages and sync secrets. You can also
# use a base Docker image here. Then we will send our Trainer class to the remote cluster with `.to()`,
# create a remote instance of the class, and instruct Runhouse to setup for PyTorch distributed training with `.distribute("pytorch")`.
# Finally, we run the remote function normally as we would locally to start the training.

if __name__ == "__main__":
    img = (
        rh.Image()
        .install_packages(
            [
                "tensorboard",
                "transformers",
                "accelerate",
                "scipy",
                "datasets",
                "s3fs",
                "torch",
            ]
        )
        .sync_secrets(["huggingface", "aws"])
    )

    num_nodes = 2
    num_gpus_per_node = 4

    cluster = rh.compute(
        name=f"rh-L4x{num_gpus_per_node}x{num_nodes}-2",
        num_nodes=num_nodes,
        instance_type=f"L4:{num_gpus_per_node}",
        provider="aws",
        image=img,
        use_spot=False,
        autostop_mins=1000,
    ).up_if_not()  # Requires access to a cloud account with the necessary permissions to launch compute.

    trainer_remote = rh.cls(BertTrainer).to(cluster, name="bert")
    trainer = trainer_remote(name="bert_trainer").distribute(
        "pytorch",
        num_replicas=num_nodes * num_gpus_per_node,
        replicas_per_node=num_gpus_per_node,
    )

    trainer.load_data(
        local_dir="./pubmed_processed",
        s3_path="s3://rh-demo-external/pubmed_processed/sample_100000",
        splits=["train", "eval", "test"],
        force_reload=True,
    )

    trainer.train()
    trainer.save_model("./final_model")
