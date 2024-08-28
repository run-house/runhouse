## You can easily test both the Airflow flow, and the underlying components and code by calling them from local

## Because the execution has been offloaded to GPU compute on remote, you can call each step from local, or from a notebook
## You can imagine that a DS or MLE might write a pipeline and interactively debug from local.
## Then, only when they are confident all the functions work, do they upload the Airflow pipeline which is minimal

## Airflow is used to schedule, monitor, and retry jobs while providing observability over runs.
## However, the code that is the substance of the program is not packed into the Airflow DAG.

import logging

from airflow_multicloud_torch_train import (
    access_data_callable,
    bring_up_cluster_callable,
    down_cluster,
    download_s3_data_callable,
    preprocess_data_callable,
    train_model_callable,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    logger.info("Starting the pipeline...")

    logger.info("Step 1: Bring up cluster")
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

    bring_up_cluster_callable(**cpu_cluster_config)

    logger.info("Step 2: Access data")
    access_data_callable(**cpu_cluster_config)

    logger.info("Step 3: Preprocess data")
    preprocess_data_callable(**cpu_cluster_config)

    logger.info("Step 4: Train model")
    bring_up_cluster_callable(**gpu_cluster_config)
    download_s3_data_callable(**gpu_cluster_config)
    train_model_callable(**gpu_cluster_config)

    logger.info("Pipeline completed.")

    down_cluster(**gpu_cluster_config)
    down_cluster(**cpu_cluster_config)
    logger.info("Cluster sucessfully downed.")
