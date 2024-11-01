# Description: This script downloads the imagenet dataset from HuggingFace, preprocesses it, and uploads it to S3.
# You don't need to reproduce this if you want to just use your own dataset directly.

import runhouse as rh


class ResNet152DataPrep:
    def __init__(self, cache_dir="~/rh_download"):
        self.ds = None
        self.cache_dir = cache_dir

    def load_data(
        self,
        dataset_name,
        train_sample="100%",
    ):
        from datasets import DatasetDict, load_dataset

        dataset = load_dataset(
            dataset_name,
            token=True,
            trust_remote_code=True,
            split=[f"train[:{train_sample}]", "validation"],
            download_mode="reuse_cache_if_exists",
            cache_dir=f"{self.cache_dir}/huggingface_cache/",
        )

        self.ds = DatasetDict(
            {
                "train": dataset[0],  # Assuming ds[0] is the train split
                "validation": dataset[1],  # Assuming ds[1] is the test split
            }
        )

    def preprocess_and_upload_data(self, save_bucket_name, save_s3_folder_prefix=""):
        from torchvision import transforms

        preprocess = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        def preprocess_example(batch):
            processed_images = []
            for img in batch["image"]:
                # Check if the image is grayscale
                if img.mode != "RGB":
                    img = img.convert("RGB")  # Convert grayscale to RGB
                processed_images.append(preprocess(img))
            batch["image"] = processed_images
            return batch

        for split_name in self.ds.keys():
            dataset = self.ds[split_name].map(preprocess_example, batched=True)
            dataset.save_to_disk(f"{self.cache_dir}/{split_name}")
            s3_path = f"s3://{save_bucket_name}/{save_s3_folder_prefix}/{split_name}"
            dataset.save_to_disk(s3_path)

        print("Uploaded Preprocessed Data")


if __name__ == "__main__":

    cluster = rh.cluster(
        name="py-preprocessing",
        instance_type="i4i.2xlarge",
        provider="aws",
        region="us-east-1",
    ).up_if_not()

    # Mount the disk to download the data to
    s3_bucket = "rh-demo-external"
    cache_dir = "/mnt/nvme"

    cluster.run(
        [
            f"sudo mkdir {cache_dir}",
            "sudo mkfs.ext4 /dev/nvme1n1",
            f"sudo mount /dev/nvme1n1 {cache_dir}",
            f"sudo chown ubuntu:ubuntu {cache_dir}",
            f"export HF_DATASETS_CACHE={cache_dir}/huggingface_cache",
            "mkdir -p $HF_DATASETS_CACHE",
        ]
    )

    env = rh.env(
        name="test_env",
        secrets=["aws", "huggingface"],
        reqs=["torch", "torchvision", "Pillow", "datasets[s3]", "s3fs"],
    )

    # Download the data, sampling down to 15% for our example
    remote_ResNet152DataPrep = rh.module(ResNet152DataPrep).to(cluster, env=env)

    dataprep = remote_ResNet152DataPrep(cache_dir=cache_dir, name="dataprep")
    dataprep.load_data(dataset_name="imagenet-1k", train_sample="1%")

    dataprep.preprocess_and_upload_data(
        save_bucket_name=s3_bucket,
        save_s3_folder_prefix="resnet-training-example/preprocessed_imagenet/",
    )
