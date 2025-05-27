# # A Basic Airflow Example Using PyTorch to Train an Image Classification NN with the MNIST Dataset

# ## Using Airflow and Kubetorch together
# This example demonstrates how to use Airflow along with Kubetorch to dispatch the work of training a basic Torch model to a remote GPU.
# The Airflow pipeline and all the callables can be run from anywhere, including local, but it will bring up remote compute with a GPU and send the training job there.

# The Torch model is identical to our [Torch training example](https://www.run.house/examples/torch-vision-mnist-basic-model-train-test), but placed within the context of an orchestrator to show how easily it is to move identically to production.
# Additionally, debugging and iteration is easy compared to debugging within a traditional orchestrator.
#
# ## The usage pattern for Kubetorch with Airflow:
# * Write Python classes and functions using normal, ordinary coding best practices. Do not think about DAGs or DSLs at all, just write great code.
# * Send the code for remote execution with Kubetorch, and figure out whether the code works, debugging it interactively. Kubetorch lets you send the code in seconds, and streams logs back. You can work on remote as if it were local.
# * Once you are satisfied with your code, you can write the callables for an Airflow. The code that is actually in the Airflow DAG is the **minimal code** to call out to already working Classes and Functions, defining the order of the steps (or you can even have a one-step Airflow DAG, making Airflow purely for scheduling and observability)
# * And you can easily iterate further on your code, or test the pipeline end-to-end from local with no Airflow participation

# Import some other libraries we need - namely Airflow, Kubetorch, and a few others.
import logging
import os
import sys
from datetime import datetime, timedelta

import kubetorch as kt
from airflow import DAG
from airflow.operators.python import PythonOperator

# ## Import the model class from the parent folder
# This class is the same as the example in https://www.run.house/examples/torch-vision-mnist-basic-model-train-test
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from torch_basic_example import SimpleTrainer

logger = logging.getLogger(__name__)

# ## Define the callable functions.
# These will be called in sequence by the Airflow PythonOperator. Each task in the Airflow DAG becomes minimal.
# These callables define both *what tasks are run* and also *where the tasks are run* - essentially, programatically controlling the dispatch.

# We can simply put the dispatch and execution of the model in the callable identical to
# how we have run it locally, ensuring identical research-to-production execution.
def run_training(**kwargs):
    compute = kt.Compute(
        gpus="1",
        image=kt.Image(image_id="nvcr.io/nvidia/pytorch:23.10-py3"),
        inactivity_ttl="0s",  # Because this is production, destroy immediately on completion
    )

    model = kt.cls(SimpleTrainer).to(compute)

    batch_size = 64
    epochs = 5
    learning_rate = 0.01

    model.load_data("./data", batch_size)

    for epoch in range(epochs):
        model.train_model(learning_rate=learning_rate)

        model.test_model()

        model.save_model(
            bucket_name="my-simple-torch-model-example",
            s3_file_path=f"checkpoints/model_epoch_{epoch + 1}.pth",
        )


# We deploy a new service for inference with the trained model checkpoint. Note that we are defining a new compute
# object rather than reusing the training compute above. Note that we load down the model weights in the image
# to achieve faster cold start times for our inference service.
def deploy_inference(**kwargs):
    logger.info("Step 2: Deploy Inference")
    checkpoint_path = "s3://my-simple-torch-model-example/checkpoints/model_final.pth"
    local_checkpoint_path = "/model.pth"
    img = kt.Image(image_id="nvcr.io/nvidia/pytorch:23.10-py3").run_bash(
        "aws s3 cp " + checkpoint_path + " " + local_checkpoint_path
    )
    inference_compute = kt.Compute(
        gpus="1",
        image=img,
    )

    init_args = dict(from_checkpoint=local_checkpoint_path)
    inference = kt.cls(SimpleTrainer).to(inference_compute, init_args=init_args)
    # We distribute the inference service as an autoscaling pool of between 0 and 6 replicas, with a maximum concurrency of 16.
    inference.distribute(num_nodes=(0, 6), max_concurrency=16)


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


deploy_inference_task = PythonOperator(
    task_id="deploy_inference_task",
    python_callable=deploy_inference,
    dag=dag,
)


# You can see that this is an incredibly minimal amount of code in Airflow. The callables are callable from the DAG. But you can also run them from a Python script, from a notebook, or anywhere else - so you can instantly iterate on the underlying classes, the functions, and by the time they run locally, they are ready for prime time in your DAG.
(bring_up_compute_task >> access_data_task >> train_model_task >> deploy_inference)
