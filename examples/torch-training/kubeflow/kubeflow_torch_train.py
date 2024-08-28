import kfp
from kfp import dsl
from kfp.components import func_to_container_op

# Define the functions for each step in the pipeline
# We can bring up an on-demand cluster using Runhouse. You can access powerful usage patterns by defining compute in code. All subsequent steps connect to this cluster by name, but you can bring up other clusters for other steps.
def bring_up_cluster(cluster_name: str, instance_type: str, provider: str):
    logger.info("Connecting to remote cluster")
    cluster = rh.ondemand_cluster(
        name="a10g-cluster", instance_type="A10G:1", provider="aws"
    ).up_if_not()

    print(cluster.is_up())
    # cluster.save() ## Use if you have a Runhouse Den account to save and monitor the resource.


def access_data(cluster_name: str, instance_type: str, provider: str):
    logger.info("Step 2: Access data")
    env = rh.env(name="test_env", reqs=["torch", "torchvision"])

    cluster = rh.cluster(name="a10g-cluster").up_if_not()
    remote_download = rh.function(download_data).to(cluster, env=env)
    remote_preprocess = rh.function(preprocess_data).to(cluster, env=env)
    logger.info("Download function sent to remote")
    remote_download()
    remote_preprocess()
    logger.info("Downloaded")


def train_model(cluster_name: str, instance_type: str, provider: str):
    logger.info("Step 3: Train Model")
    cluster = rh.cluster(name="a10g-cluster").up_if_not()

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


def down_cluster(cluster_name: str, instance_type: str, provider: str):
    cluster = rh.cluster(name="a10g-cluster")
    cluster.teardown()


# Convert the functions to Kubeflow Pipelines components
bring_up_cluster_op = func_to_container_op(bring_up_cluster)
access_data_op = func_to_container_op(access_data)
preprocess_data_op = func_to_container_op(preprocess_data)
download_s3_data_op = func_to_container_op(download_s3_data)
train_model_op = func_to_container_op(train_model)
down_cluster_op = func_to_container_op(down_cluster)

# Define the pipeline
@dsl.pipeline(
    name="PyTorch Training Pipeline",
    description="A simple PyTorch training pipeline with multiple steps",
)
def pytorch_training_pipeline(cluster_name: str, instance_type: str, provider: str):
    bring_up_cluster_task = bring_up_cluster_op(cluster_name, instance_type, provider)
    access_data_task = access_data_op(cluster_name, instance_type, provider).after(
        bring_up_cluster_task
    )
    preprocess_data_task = preprocess_data_op(
        cluster_name, instance_type, provider
    ).after(access_data_task)
    download_s3_data_task = download_s3_data_op(
        cluster_name, instance_type, provider
    ).after(preprocess_data_task)
    train_model_task = train_model_op(cluster_name, instance_type, provider).after(
        download_s3_data_task
    )
    down_cluster_task = down_cluster_op(cluster_name, instance_type, provider).after(
        train_model_task
    )
    _ = down_cluster_task


# Compile the pipeline
kfp.compiler.Compiler().compile(
    pytorch_training_pipeline, "pytorch_training_pipeline.yaml"
)
