# # A Basic Airflow Example Using PyTorch to Train an Image Classification NN with the MNIST Dataset

# ## Using Airflow and Runhouse together
# This example demonstrates how to use Airflow along with Runhouse to dispatch the work of training a basic Torch model to a remote GPU.
# The Airflow pipeline and all the callables can be run from anywhere, including local, but it will bring up an cluster on AWS with a GPU and send the training job there.

# The Torch model is identical to our [Torch training example](https://www.run.house/examples/torch-vision-mnist-basic-model-train-test), but placed within the context of an orchestrator to show how Runhouse allows for more flexible
# debugging and dispatch when used in conjunction with a traditional orchestrator. The model code uses the very popular MNIST dataset which includes a large number of handwritten digits, and create a neural network that accurately identifies
# what digit is in an image.
#
# ## The usage pattern for Runhouse with Airflow:
# * Write Python classes and functions using normal, ordinary coding best practices. Do not think about DAGs or DSLs at all, just write great code.
# * Send the code for remote execution with Runhouse, and figure out whether the code works, debugging it interactively. Runhouse lets you send the code in seconds, and streams logs back. You can work on remote as if it were local.
# * Once you are satisfied with your code, you can write the callables for an Airflow PythonOperator. The code that is actually in the Airflow DAG is the **minimal code** to call out to already working Classes and Functions, defining the order of the steps (or you can even have a one-step Airflow DAG, making Airflow purely for scheduling and observability)
# * And you can easily iterate further on your code, or test the pipeline end-to-end from local with no Airflow participation

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
from TorchBasicExample import download_data, preprocess_data, SimpleTrainer

logger = logging.getLogger(__name__)

# ## Define the callable functions.
# These will be called in sequence by the Airflow PythonOperator. Each task in the Airflow DAG becomes minimal.
# These callables define both *what tasks are run* and also *where the tasks are run* - essentially, programatically controlling the dispatch.

# We can bring up an on-demand cluster using Runhouse. You can access powerful usage patterns by defining compute in code. All subsequent steps connect to this cluster by name, but you can bring up other clusters for other steps.
def bring_up_cluster_callable(**kwargs):
    logger.info("Connecting to remote cluster")
    img = rh.Image().install_packages(["torch", "torchvision"])
    cluster = rh.compute(
        name="a10g-cluster", instance_type="g5.xlarge", provider="aws", image=img
    ).up_if_not()

    print(cluster.is_up())
    # cluster.save() ## Use if you have a Runhouse Den account to save and monitor the resource.


# We will send the function to download data to the remote cluster and then invoke it to download the data to the remote machine. You can imagine that this is a data access or pre-processing step after which data is prepared.
def access_data_callable(**kwargs):
    logger.info("Step 2: Access data")
    cluster = rh.compute(name="a10g-cluster").up_if_not()
    remote_download = rh.function(download_data).to(cluster)
    remote_preprocess = rh.function(preprocess_data).to(cluster)
    logger.info("Download function sent to remote")
    remote_download()
    remote_preprocess()
    logger.info("Downloaded")


# Then we instantiate the trainer, and then invoke the training on the remote machine. On the remote, we have a GPU. This is also a natural point to split the workflow if we want to do some tasks on GPU and some on CPU.
def train_model_callable(**kwargs):
    logger.info("Step 3: Train Model")
    cluster = rh.compute(name="a10g-cluster").up_if_not()

    remote_torch_example = rh.cls(SimpleTrainer).to(
        cluster, name="torch-basic-training"
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


# We programatically down the cluster, but we can also reuse this cluster by name.
def down_cluster(**kwargs):
    cluster = rh.compute(name="a10g-cluster")
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

dag = DAG(
    "pytorch_training_pipeline_example",
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
    dag=dag,
)

access_data_task = PythonOperator(
    task_id="access_data_task",
    python_callable=access_data_callable,
    dag=dag,
)


train_model_task = PythonOperator(
    task_id="train_model_task",
    python_callable=train_model_callable,
    dag=dag,
)


down_cluster_task = PythonOperator(
    task_id="down_cluster_task",
    python_callable=down_cluster,
    dag=dag,
)


# You can see that this is an incredibly minimal amount of code in Airflow. The callables are callable from the DAG. But you can also run them from a Python script, from a notebook, or anywhere else - so you can instantly iterate on the underlying classes, the functions, and by the time they run locally, they are ready for prime time in your DAG.
(
    run_sky_status
    >> bring_up_cluster_task
    >> access_data_task
    >> train_model_task
    >> down_cluster_task
)
