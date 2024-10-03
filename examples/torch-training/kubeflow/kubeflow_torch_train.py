import kfp
from kfp import kubernetes
from kfp.dsl import component, pipeline

# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
# sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# from TorchBasicExample import download_data, preprocess_data, SimpleTrainer

# Define the functions for each step in the pipeline
# We can bring up an on-demand cluster using Runhouse. You can access powerful usage patterns by defining compute in code. All subsequent steps connect to this cluster by name, but you can bring up other clusters for other steps.
@component(base_image="pypypypy/my-pipeline-image:latest")
def bring_up_cluster(cluster_name: str, instance_type: str):

    # First we configure the environment to setup Runhouse and AWS credentials. We only need to configure the AWS credentials in the first step since the cluster is saved to Runhouse and we reuse the resource.
    import os

    import runhouse as rh

    rh.login(token=os.getenv("RUNHOUSE_API_KEY"))

    import subprocess

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
    cluster = rh.cluster(
        name="/paul/" + cluster_name
    ).up_if_not()  # I am adding /paul/ since I have saved the cluster to Runhouse with my account.

    # Grab the functions from the TorchBasicExample module and send them to the remote cluster
    sys.path.append(os.path.expanduser("~/training"))
    from TorchBasicExample import download_data, preprocess_data

    remote_download = rh.function(download_data).to(cluster, env=env)
    remote_preprocess = rh.function(preprocess_data).to(cluster, env=env)

    # Run the data download
    remote_download()
    remote_preprocess("./data")


# Now we run the training.
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


@component(base_image="pypypypy/my-pipeline-image:latest")
def down_cluster(cluster_name: str):
    import os

    import runhouse as rh

    rh.login(token=os.getenv("RUNHOUSE_API_KEY"))

    cluster = rh.cluster(name="/paul/" + cluster_name)
    cluster.teardown()


# Define the pipeline
@pipeline(
    name="PyTorch Training Pipeline",
    description="A simple PyTorch training pipeline with multiple steps",
)
def pytorch_training_pipeline(cluster_name: str, instance_type: str):
    # Define tasks from the components
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

    # Define the execution order
    access_data_task.after(bring_up_cluster_task)
    train_model_task.after(access_data_task)
    down_cluster_task.after(train_model_task)


# Compile the pipeline
kfp.compiler.Compiler().compile(
    pytorch_training_pipeline, "pytorch_training_pipeline.yaml"
)
