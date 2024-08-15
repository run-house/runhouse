## You can easily test both the Airflow flow, and the underlying components and code by calling them from local

## Because the execution has been offloaded to GPU compute on remote, you can call each step from local, or from a notebook
## You can imagine that a DS or MLE might write a pipeline and interactively debug from local.
## Then, only when they are confident all the functions work, do they upload the Airflow pipeline which is minimal

## Airflow is used to schedule, monitor, and retry jobs while providing observability over runs.
## However, the code that is the substance of the program is not packed into the Airflow DAG.

import logging

from callables import (
    access_data_callable,
    bring_up_cluster_callable,
    down_cluster,
    train_model_callable,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    logger.info("Starting the pipeline...")

    logger.info("Step 1: Bring up cluster")
    bring_up_cluster_callable()

    logger.info("Step 2: Access data")
    access_data_callable()

    logger.info("Step 3: Train model")
    train_model_callable()

    logger.info("Pipeline completed.")

    down_cluster()
    logger.info("Cluster sucessfully downed.")
