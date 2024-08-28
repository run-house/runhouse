# ## Using Multi-Cloud with Airflow

# This basic Airflow example combines CPU processing on AWS, storage on S3, and GPU processing on Google Cloud.
# There are several reasons why you might want to use multiple clouds in a single workflow:
# * Your data lives on AWS, but you are quota limited on GPUs there or don't have access to powerful GPUs. You can do pre-processing on AWS and then train on a second cloud where you have GPUs.
# * You don't want GPUs to sit idle while you are doing pre-processing or post-processing. You can use a GPU cluster on-demand and then down it when you are done.
# * You want to right-size instances for each step of execution rather than bringing up a box that is the maximum of each of your memory, CPU, and GPU requirements.

# The Torch model is identical to our [Torch training example](https://www.run.house/examples/torch-vision-mnist-basic-model-train-test).
#
# ## The usage pattern for Runhouse with Airflow:
# * Write Python classes and functions using normal, ordinary coding best practices. Do not think about DAGs or DSLs at all, just write great code.
# * Send the code for remote execution with Runhouse, and figure out whether the code works, debugging it interactively. Runhouse lets you send the code in seconds, and streams logs back. You can work on remote as if it were local.
# * Once you are satisfied with your code, you can write the callables for an Airflow PythonOperator. The code that is actually in the Airflow DAG is the **minimal code** to call out to already working Classes and Functions, defining the order of the steps (or you can even have a one-step Airflow DAG, making Airflow purely for scheduling and observability)
# * And you can easily iterate further on your code, or test the pipeline end-to-end from local with no Airflow participation

# ## Setup credentials and dependencies
#
# Optionally, set up a virtual environment:
# ```shell
# $ conda create -n demo-runhouse python=3.10
# $ conda activate demo-runhouse
# ```
# Install the required dependencies:
# ```shell
# $ pip install "runhouse[aws]" torch torchvision airflow
# ```
#
# We'll be launching an AWS EC2 GCP instance via [SkyPilot](https://github.com/skypilot-org/skypilot), so we need to
# make sure our AWS credentials are set up:
# ```shell
# $ aws configure
# $ gcloud init
# $ gcloud auth application-default login
# $ sky check
# ```
#

# Import some other libraries we need - namely Airflow, Runhouse, and a few others.
import logging
import os

# ## Import the model class from the parent folder
# This class is the same as the example in https://www.run.house/examples/torch-vision-mnist-basic-model-train-test
import sys
from datetime import datetime, timedelta

import runhouse as rh
from airflow import DAG
from airflow.operators.bash_operator import BashOperator
from airflow.operators.python import PythonOperator

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from DataProcessing import download_folder_from_s3, upload_folder_to_s3
from TorchBasicExample import download_data, preprocess_data, SimpleTrainer


logger = logging.getLogger(__name__)

cpu_cluster_name = "cpu-cluster"  # To be used later to name the clusters we raise
gpu_cluster_name = "gpu-cluster"


def get_cluster(**kwargs):
    return rh.cluster(
        name=kwargs.get("cluster_name", "rh-cluster"),
        instance_type=kwargs.get("instance_type"),
        provider=kwargs.get("provider", "aws"),
    ).up_if_not()


# ## Define the callable functions.
# These will be called in sequence by the Airflow PythonOperator. Each task in the Airflow DAG becomes minimal.
# These callables define both *what tasks are run* and also *where the tasks are run* - essentially, programatically controlling the dispatch.

# We can bring up an on-demand cluster using Runhouse. You can access powerful usage patterns by defining compute in code. All subsequent steps connect to this cluster by name, but you can bring up other clusters for other steps.
def bring_up_cluster_callable(**kwargs):
    logger.info("Connecting to remote cluster")
    cluster = get_cluster(**kwargs)
    print(cluster.is_up())

    # cluster.save() ## Use if you have a Runhouse Den account to save and monitor the resource.


# We will send the function to download data to the remote cluster and then invoke it to download the data to the remote machine. You can imagine that this is a data access or pre-processing step after which data is prepared.
def access_data_callable(**kwargs):
    logger.info("Step 3: Preprocess the Data")

    env = rh.env(name="test_env", reqs=["torch", "torchvision"])

    cluster = get_cluster(**kwargs)

    remote_download = rh.function(download_data).to(cluster, env=env)
    logger.info("Download function sent to remote")
    remote_download()
    logger.info("Data downloaded")


# Now execute the preprocessing on the CPU-only cluster
def preprocess_data_callable(**kwargs):
    cluster = get_cluster(**kwargs)

    env = rh.env(name="test_env", secrets=["aws"], reqs=["torch", "torchvision"])

    remote_preprocess_data = rh.function(preprocess_data).to(cluster, env=env)
    remote_upload = rh.function(upload_folder_to_s3).to(cluster, env=env)
    logger.info("Data preprocessing and upload functions sent to cluster")

    remote_preprocess_data("./data")
    logger.info("Data preprocessed")
    remote_upload("./data", "rh-demo-external", "torch-training-example")

    logger.info("Saved to S3")


# Download the data from S3. This is runnable from anywhere (including local), but also on a newly launched instance
def download_s3_data_callable(**kwargs):
    cluster = get_cluster(**kwargs)

    s3_download = rh.function(download_folder_from_s3).to(cluster)
    s3_download("rh-demo-external", "torch-training-example", "./data")


# Then we instantiate the trainer, and then invoke the training on the remote machine. On the remote, we have a GPU. This is also a natural point to split the workflow if we want to do some tasks on GPU and some on CPU.
def train_model_callable(**kwargs):
    logger.info("Step 4: Train Model")
    cluster = get_cluster(**kwargs)

    env = rh.env(name="test_env", reqs=["torch", "torchvision"])

    remote_torch_example = rh.module(SimpleTrainer).to(
        cluster, env=env, name="torch-basic-training"
    )

    model = remote_torch_example()

    batch_size = 64
    epochs = 5
    learning_rate = 0.01
    cluster.run(["ls"])
    model.load_train("./data", batch_size)
    model.load_test("./data", batch_size)

    for epoch in range(epochs):
        model.train_model(learning_rate=learning_rate)
        model.test_model()
        model.save_model(
            bucket_name="my-simple-torch-model-example",
            s3_file_path=f"checkpoints/model_epoch_{epoch + 1}.pth",
        )


# We programatically down the cluster, but we can also reuse this cluster by name.
def down_cluster(**kwargs):
    cluster = get_cluster(**kwargs)
    cluster.teardown()


# ## Define the Airflow DAG
# This is a simple DAG with multiple steps. Each step is a PythonOperator that calls a function defined above.
default_args = {
    "owner": "paul",
    "depends_on_past": False,
    "start_date": datetime(2024, 8, 6),
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}
cpu_cluster_config = {
    "cluster_name": "cpu-cluster",
    "instance_type": "CPU:4+",
    "provider": "aws",
}
gpu_cluster_config = {
    "cluster_name": "gpu-cluster",
    "instance_type": "L4:1",
    "provider": "gcp",
}


dag = DAG(
    "pytorch_training_pipeline_example_multicloud",
    default_args=default_args,
    description="A simple PyTorch training DAG with multiple steps",
    schedule=timedelta(days=1),
)
run_sky_status = BashOperator(
    task_id="run_sky_status",
    bash_command="sky status",
    dag=dag,
)

bring_up_cluster_task = PythonOperator(
    task_id="bring_up_cluster_task",
    python_callable=bring_up_cluster_callable,
    op_kwargs=cpu_cluster_config,
    dag=dag,
)

access_data_task = PythonOperator(
    task_id="access_data_task",
    python_callable=access_data_callable,
    op_kwargs=cpu_cluster_config,
    dag=dag,
)

preprocess_data_task = PythonOperator(
    task_id="preprocess_data_task",
    python_callable=preprocess_data_callable,
    op_kwargs=cpu_cluster_config,
    dag=dag,
)

bring_up_gpu_cluster_task = PythonOperator(
    task_id="bring_up_gpu_cluster_task",
    python_callable=bring_up_cluster_callable,
    op_kwargs=gpu_cluster_config,
    dag=dag,
)

download_data_from_s3_task = PythonOperator(
    task_id="download_data_from_s3_task",
    python_callable=download_s3_data_callable,
    op_kwargs=gpu_cluster_config,
    dag=dag,
)

train_model_task = PythonOperator(
    task_id="train_model_task",
    python_callable=train_model_callable,
    op_kwargs=gpu_cluster_config,
    dag=dag,
)

down_cluster_task = PythonOperator(
    task_id="down_cluster_task",
    python_callable=down_cluster,
    op_kwargs=cpu_cluster_config,
    dag=dag,
)

down_gpu_cluster_task = PythonOperator(
    task_id="down_gpu_cluster_task",
    python_callable=down_cluster,
    op_kwargs=gpu_cluster_config,
    dag=dag,
)


# You can see that this is an incredibly minimal amount of code in Airflow. The callables are callable from the DAG. But you can also run them from a Python script, from a notebook, or anywhere else - so you can instantly iterate on the underlying classes, the functions, and by the time they run locally, they are ready for prime time in your DAG.
(
    run_sky_status
    >> bring_up_cluster_task
    >> access_data_task
    >> preprocess_data_task
    >> bring_up_gpu_cluster_task
    >> download_data_from_s3_task
    >> train_model_task
    >> down_cluster_task
    >> down_gpu_cluster_task
)
