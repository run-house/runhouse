# Description: This script downloads the imagenet dataset from HuggingFace, preprocesses it, and uploads it to S3.
# You don't need to reproduce this if you want to just use your own dataset directly.

import runhouse as rh


def download_data(dataset_name, train_sample, save_bucket_name, save_s3_folder_prefix):
    from datasets import load_dataset

    ds = load_dataset(
        dataset_name,
        token=True,
        trust_remote_code=True,
        split=[f"train[:{train_sample}]", "test"],
        download_mode="reuse_cache_if_exists",
        cache_dir="/mnt/nvme/huggingface_cache",
    )
    s3_path = f"s3://{save_bucket_name}/{save_s3_folder_prefix}"
    ds.save_to_disk(s3_path)

    print("Downloaded Data")


def preprocess_data(load_path, save_bucket_name, save_s3_folder_prefix=""):
    from datasets import load_dataset
    from torchvision import transforms

    ds = load_dataset(load_path)  # "/mnt/nvme/downloaded_image_net"

    preprocess = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
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

    for split_name in ds.keys():
        if split_name == "test":
            dataset = ds[split_name].map(preprocess_example, batched=True)
            print("Preprocessed Dataset")
            dataset.save_to_disk(f"/mnt/nvme/preprocessed_imagenet/{split_name}")
            s3_path = f"s3://{save_bucket_name}/{save_s3_folder_prefix}/{split_name}"
            ds.save_to_disk(s3_path)

    print("Uploaded Preprocessed Data")


def send_data_to_s3(bucket_name, s3_folder_prefix, local_folder_path):
    import os

    import boto3

    s3 = boto3.client("s3")

    for root, dirs, files in os.walk(local_folder_path):
        for file in files:
            local_path = os.path.join(root, file)
            relative_path = os.path.relpath(local_path, local_folder_path)
            s3_path = os.path.join(s3_folder_prefix, relative_path)

            s3.upload_file(local_path, bucket_name, s3_path)
            print(f"Uploaded {local_path} to s3://{bucket_name}/{s3_path}")


if __name__ == "__main__":

    cluster = rh.cluster(
        name="py-preprocessing",
        instance_type="i4i.2xlarge",
        provider="aws",
        region="us-east-1",
    ).up_if_not()

    #
    cluster.run(
        [
            "sudo mkdir /mnt/nvme",
            "sudo mkfs.ext4 /dev/nvme1n1",
            "sudo mount /dev/nvme1n1 /mnt/nvme",
            "sudo chown ubuntu:ubuntu /mnt/nvme",
            "export HF_DATASETS_CACHE=/mnt/nvme/huggingface_cache",
            "mkdir -p $HF_DATASETS_CACHE",
        ]
    )

    env = rh.env(
        name="test_env",
        secrets=["aws", "huggingface"],
        reqs=["torch", "torchvision", "Pillow", "datasets[s3]", "boto3"],
    )
    s3_bucket = "rh-demo-external"

    # Download the data, sampling down to 5% for our example
    remote_download_data = rh.function(download_data).to(cluster, env=env)
    remote_download_data(
        dataset_name="imagenet-1k",
        save_bucket_name=s3_bucket,
        save_s3_folder_prefix="resnet-training-example/downloaded_image_net/",
        train_sample="15%",
    )

    # Preprocess the data and upload to S3
    remote_preprocess_data = rh.function(preprocess_data).to(cluster, env=env)
    remote_preprocess_data(
        load_path=s3_bucket + "/resnet-training-example/downloaded_image_net",
        save_bucket_name=s3_bucket,
        save_s3_folder_prefix="resnet-training-example/preprocessed_imagenet/",
    )
