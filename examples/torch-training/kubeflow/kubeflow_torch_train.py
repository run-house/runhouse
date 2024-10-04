# # A Kubeflow Training Pipeline with Runhouse

# This example demonstrates how to use Kubeflow to orchestrate the training of a basic Torch model,
# with the training class dispatched to a GPU-enabled AWS cloud instance to actually do the training.
#
# We use the very popular MNIST dataset which includes a large number
# of handwritten digits, and create a neural network that accurately identifies
# what digit is in an image.
#
# ## Setup credentials and dependencies
# For this example, we will need AWS cloud credentials and a Runhouse API key. We name this secret `my-secret` for simplicity, and use it in the pipeline.
# kubectl create secret generic my-secret \
#   --from-literal=AWS_ACCESS_KEY_ID=<your-access-key-id> \
#   --from-literal=AWS_SECRET_ACCESS_KEY=<your-secret-access-key> \
#   --from-literal=RUNHOUSE_API_KEY=<your-runhouse-api-key>
#
# We'll be launching elastic compute from AWS from within the first step, but you can use any compute resource. You can see in the multi-cloud example that you can even run steps on different clusters - for instance, run CPU pre-processing on the cluster that hosts Kubeflow, while offloading GPU to an elastic instance.
#
# ## Setting up the Kubeflow pipeline
# The Kubeflow pipeline is extremely lean, with each step being a function that is run on a remote cluster. The functions are defined in the TorchBasicExample module, and are sent to the remote cluster using Runhouse.

import kfp
from kfp import kubernetes
from kfp.dsl import component, pipeline

# First, we can bring up an on-demand cluster using Runhouse. You can access powerful usage patterns by defining compute in code. All subsequent steps connect to this cluster by name, but you can also bring up other clusters for other steps.
@component(base_image="pypypypy/my-pipeline-image:latest")
def bring_up_cluster(cluster_name: str, instance_type: str):

    # First we configure the environment to setup Runhouse and AWS credentials. We only need to configure the AWS credentials in the first step since the cluster is saved to Runhouse and we reuse the resource.
    import os
    import subprocess

    import runhouse as rh

    rh.login(token=os.getenv("RUNHOUSE_API_KEY"))

    subprocess.run(
        [
            "aws",
            "configure",
            "set",
            "aws_access_key_id",
            os.getenv("AWS_ACCESS_KEY_ID"),
        ],
        check=True,
    )
    subprocess.run(
        [
            "aws",
            "configure",
            "set",
            "aws_secret_access_key",
            os.getenv("AWS_SECRET_ACCESS_KEY"),
        ],
        check=True,
    )

    # Now we bring up the cluster and save it to Runhouse to reuse in subsequent steps. This allows for reuse of the same compute and much better statefulness across multiple Kubeflow steps.
    cluster = rh.ondemand_cluster(
        name=cluster_name,
        instance_type=instance_type,
        provider="aws",
        autostop_mins=120,
    ).up_if_not()

    print(cluster.is_up())
    cluster.save()


# This step represents a step to access and lightly preprocess the dataset. The MNIST example is trivial, but we are doing this preprocessing on the same compute we will use later to do the training.
@component(base_image="pypypypy/my-pipeline-image:latest")
def access_data(cluster_name: str):
    import os
    import sys

    import runhouse as rh

    rh.login(token=os.getenv("RUNHOUSE_API_KEY"))
    # Define the requirements on the remote cluster itself, and send the functions which are on the Docker image to the remote cluster
    env = rh.env(name="test_env", reqs=["torch", "torchvision"])
    cluster = rh.cluster(name="/paul/" + cluster_name).up_if_not()

    # Grab the functions from the TorchBasicExample module and send them to the remote cluster
    sys.path.append(os.path.expanduser("~/training"))
    from TorchBasicExample import download_data, preprocess_data

    remote_download = rh.function(download_data).to(cluster, env=env)
    remote_preprocess = rh.function(preprocess_data).to(cluster, env=env)

    # Run the data download
    remote_download()
    remote_preprocess("./data")


# Now we run the training. In this step, we dispatch the training to the remote cluster. The model is trained on the remote cluster, and the model checkpoints are saved to an S3 bucket.
@component(base_image="pypypypy/my-pipeline-image:latest")
def train_model(cluster_name: str):
    import os
    import sys

    import runhouse as rh

    rh.login(token=os.getenv("RUNHOUSE_API_KEY"))
    cluster = rh.cluster(name="/paul/" + cluster_name).up_if_not()
    env = rh.env(name="test_env", reqs=["torch", "torchvision"])

    # Grab the Trainer class from the TorchBasicExample module and send it to the remote cluster
    sys.path.append(os.path.expanduser("~/training"))
    from TorchBasicExample import SimpleTrainer

    remote_torch_example = rh.module(SimpleTrainer).to(
        cluster, env=env, name="torch-basic-training"
    )

    batch_size = 64
    epochs = 5
    learning_rate = 0.01

    # Instantiate the model (remotely) and run the training loop
    model = remote_torch_example()
    model.load_train("./data", batch_size)
    model.load_test("./data", batch_size)

    for epoch in range(epochs):
        model.train_model(learning_rate=learning_rate)
        model.test_model()
        model.save_model(
            bucket_name="my-simple-torch-model-example",
            s3_file_path=f"checkpoints/model_epoch_{epoch + 1}.pth",
        )


# Finally, we can down the cluster after the training is done.
@component(base_image="pypypypy/my-pipeline-image:latest")
def down_cluster(cluster_name: str):
    import os

    import runhouse as rh

    rh.login(token=os.getenv("RUNHOUSE_API_KEY"))

    cluster = rh.cluster(name="/paul/" + cluster_name)
    cluster.teardown()


# Define the pipeline. This is a simple linear pipeline with four steps: bring up the cluster, access the data, train the model, and down the cluster.
@pipeline(
    name="PyTorch Training Pipeline",
    description="A simple PyTorch training pipeline with multiple steps",
)
def pytorch_training_pipeline(cluster_name: str, instance_type: str):
    bring_up_cluster_task = bring_up_cluster(
        cluster_name=cluster_name, instance_type=instance_type
    )
    access_data_task = access_data(cluster_name=cluster_name)
    train_model_task = train_model(cluster_name=cluster_name)
    down_cluster_task = down_cluster(cluster_name=cluster_name)

    # Provide Kubernetes secrets to the tasks
    env_vars = ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "RUNHOUSE_API_KEY"]
    tasks = [
        bring_up_cluster_task,
        access_data_task,
        train_model_task,
        down_cluster_task,
    ]
    secret_name = "my-secret"  # This is the name of your secret in Kubernetes secrets
    for var in env_vars:
        for task in tasks:
            kubernetes.use_secret_as_env(
                task,
                secret_name=secret_name,
                secret_key_to_env={var: var.upper()},
            )

    access_data_task.after(bring_up_cluster_task)
    train_model_task.after(access_data_task)
    down_cluster_task.after(train_model_task)


# Compile the pipeline
kfp.compiler.Compiler().compile(
    pytorch_training_pipeline, "pytorch_training_pipeline.yaml"
)
