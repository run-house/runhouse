from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash_operator import BashOperator
from airflow.operators.python import PythonOperator

# Import the functions from the training script
from callables import (
    access_data_callable,
    bring_up_cluster_callable,
    down_cluster,
    train_model_callable,
)

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


(
    run_sky_status
    >> bring_up_cluster_task
    >> access_data_task
    >> train_model_task
    >> down_cluster_task
)
