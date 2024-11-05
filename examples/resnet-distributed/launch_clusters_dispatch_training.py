import asyncio

import runhouse as rh
from resnet152_trainer import ResNet152Trainer


async def start_training(
    cluster,
    num_nodes,
    gpus_per_node,
    working_s3_bucket,
    working_s3_path,
    train_data_path,
    val_data_path,
    epochs,
):
    train_workers = []
    for i in range(num_nodes):
        for j in range(gpus_per_node):
            # Per https://pytorch.org/docs/stable/distributed.html#environment-variable-initialization
            dist_config = {
                "MASTER_ADDR": cluster.internal_ips[0],
                "MASTER_PORT": "12345",
                "RANK": str(i * gpus_per_node + j),
                "WORLD_SIZE": str(num_nodes * gpus_per_node),
            }

            env = rh.env(
                name=f"pytorch_env_{i}",
                reqs=[
                    "torch",
                    "torchvision",
                    "Pillow",
                    "datasets[s3]",
                    "boto3",
                    "s3fs",
                ],
                secrets=["aws", "huggingface"],
                env_vars=dist_config,
                compute={"node_idx": i},
            )
            # While iterating, you can kill the worker processes to stop any pending or hanging calls
            # cluster.delete(env.name)
            train_worker = rh.module(ResNet152Trainer).to(
                cluster, env=env, name=f"train_{i}"
            )
            remote_train_worker = train_worker(
                s3_bucket=working_s3_bucket, s3_path=working_s3_path
            )
            train_workers.append(remote_train_worker)

    await asyncio.gather(
        *[
            train_worker.train(
                num_epochs=epochs,
                num_classes=1000,
                train_data_path=train_data_path,
                val_data_path=val_data_path,
                run_async=True,
            )
            for train_worker in train_workers
        ]
    )


if __name__ == "__main__":
    # Set up the s3 buckets we will use for training
    train_data_path = (
        "s3://rh-demo-external/resnet-training-example/preprocessed_imagenet/train/"
    )
    val_data_path = "s3://rh-demo-external/resnet-training-example/preprocessed_imagenet/validation/"

    working_s3_bucket = "rh-demo-external"
    working_s3_path = "resnet-training-example/"

    # Create a cluster of 3 GPUs
    gpus_per_node = 1
    num_nodes = 3
    gpu_cluster_name = f"py-{num_nodes}x{gpus_per_node}GPU"

    gpu_cluster = rh.cluster(
        name=gpu_cluster_name,
        instance_type=f"A10G:{gpus_per_node}",
        num_instances=num_nodes,
        provider="aws",
    ).up_if_not()

    gpu_cluster.restart_server()

    epochs = 15
    asyncio.run(
        start_training(
            gpu_cluster,
            num_nodes=num_nodes,
            gpus_per_node=gpus_per_node,
            working_s3_bucket=working_s3_bucket,
            working_s3_path=working_s3_path,
            train_data_path=train_data_path,
            val_data_path=val_data_path,
            epochs=epochs,
        )
    )

    gpu_cluster.teardown()
