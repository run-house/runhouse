import os
import sys

import kfp
from kfp.dsl import component, pipeline

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
import logging

from TorchBasicExample import download_data, preprocess_data, SimpleTrainer


# Define the functions for each step in the pipeline
# We can bring up an on-demand cluster using Runhouse. You can access powerful usage patterns by defining compute in code. All subsequent steps connect to this cluster by name, but you can bring up other clusters for other steps.
logger = logging.getLogger(__name__)


@component
def bring_up_cluster(cluster_name: str, instance_type: str, provider: str):
    import runhouse as rh

    logger.info("Connecting to remote cluster")
    cluster = rh.ondemand_cluster(
        name=cluster_name, instance_type=instance_type, provider=provider
    ).up_if_not()

    print(cluster.is_up())
    # cluster.save() ## Use if you have a Runhouse Den account to save and monitor the resource.


@component
def access_data(cluster_name: str, instance_type: str, provider: str):
    import runhouse as rh

    logger.info("Step 2: Access data")
    env = rh.env(name="test_env", reqs=["torch", "torchvision"])

    cluster = rh.cluster(name=cluster_name).up_if_not()
    remote_download = rh.function(download_data).to(cluster, env=env)
    remote_preprocess = rh.function(preprocess_data).to(cluster, env=env)
    logger.info("Download function sent to remote")
    remote_download()
    remote_preprocess()
    logger.info("Downloaded")


@component
def train_model(cluster_name: str, instance_type: str, provider: str):
    import runhouse as rh

    logger.info("Step 3: Train Model")
    cluster = rh.cluster(name=cluster_name).up_if_not()

    env = rh.env(name="test_env", reqs=["torch", "torchvision"])

    remote_torch_example = rh.module(SimpleTrainer).to(
        cluster, env=env, name="torch-basic-training"
    )

    model = remote_torch_example()

    batch_size = 64
    epochs = 5
    learning_rate = 0.01

    model.load_train("./data", batch_size)
    model.load_test("./data", batch_size)

    for epoch in range(epochs):
        model.train_model(learning_rate=learning_rate)
        model.test_model()
        model.save_model(
            bucket_name="my-simple-torch-model-example",
            s3_file_path=f"checkpoints/model_epoch_{epoch + 1}.pth",
        )


@component
def down_cluster(cluster_name: str, instance_type: str, provider: str):
    import runhouse as rh

    cluster = rh.cluster(name=cluster_name)
    cluster.teardown()


# Define the pipeline
@pipeline(
    name="PyTorch Training Pipeline",
    description="A simple PyTorch training pipeline with multiple steps",
)
def pytorch_training_pipeline(cluster_name: str, instance_type: str, provider: str):
    bring_up_cluster_task = bring_up_cluster(
        cluster_name=cluster_name, instance_type=instance_type, provider=provider
    )
    access_data_task = access_data(
        cluster_name=cluster_name, instance_type=instance_type, provider=provider
    )
    train_model_task = train_model(
        cluster_name=cluster_name, instance_type=instance_type, provider=provider
    )
    down_cluster_task = down_cluster(
        cluster_name=cluster_name, instance_type=instance_type, provider=provider
    )

    # Define the execution order
    access_data_task.after(bring_up_cluster_task)
    train_model_task.after(access_data_task)
    down_cluster_task.after(train_model_task)


# Compile the pipeline
kfp.compiler.Compiler().compile(
    pytorch_training_pipeline, "pytorch_training_pipeline.yaml"
)
