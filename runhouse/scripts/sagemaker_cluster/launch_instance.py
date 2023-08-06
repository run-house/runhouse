"""Script used to keep the SageMaker cluster up, pending the autostop time provided in the cluster's config."""
import logging
import os
import subprocess
import time

import yaml

logger = logging.getLogger(__name__)

DEFAULT_AUTOSTOP = -1

# https://github.com/aws-samples/sagemaker-ssh-helper#step-3-modify-your-training-script
import sagemaker_ssh_helper

sagemaker_ssh_helper.setup_and_start_ssh()


def run_training_job(path_to_job, num_attempts):
    job_succeeded = False

    try:
        # Execute the script as a separate process and capture stdout and stderr
        completed_process = subprocess.run(
            ["python", path_to_job],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        # Access stdout and stderr
        stdout = completed_process.stdout
        stderr = completed_process.stderr

        if stdout:
            print(f"Stdout\n: {stdout}")

        if stderr:
            print(f"stderr\n: {stderr}")

        job_succeeded = True

    except subprocess.CalledProcessError as e:
        logger.error(
            f"({e.returncode}) Error executing training script "
            f"(already made {num_attempts} attempts): {e.stderr}"
        )
        num_attempts += 1
    except Exception as e:
        num_attempts += 1
        logger.error(
            f"Error executing training script (already made {num_attempts} attempts): {e}"
        )

    finally:
        return job_succeeded, num_attempts


def read_cluster_config():
    try:
        # Read the autostop from the cluster's config
        with open(os.path.expanduser("~/.rh/cluster_config.yaml"), "r") as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        config = {}

    return config


if __name__ == "__main__":
    launched_time = time.time()
    last_autostop_value = None
    training_job_completed = False
    path_to_job = None
    num_attempts = 0

    while True:
        config = read_cluster_config()

        autostop = int(config.get("autostop_mins", DEFAULT_AUTOSTOP))

        if autostop != -1:
            time_to_autostop = launched_time + (autostop * 60)
            current_time = time.time()

            if current_time >= time_to_autostop:
                print("Autostop time reached, stopping instance")
                break

            # Reset launch time if autostop was updated
            if last_autostop_value is not None and autostop != last_autostop_value:
                print(f"Resetting autostop from {last_autostop_value} to {autostop}")
                launched_time = current_time

            last_autostop_value = autostop

        estimator_entry_point = config.get("estimator_entry_point")
        estimator_source_dir = config.get("estimator_source_dir")

        if estimator_source_dir:
            # Update the path to the custom estimator job on the cluster
            folder_name = os.path.basename(estimator_source_dir)
            path_to_job = f"/opt/ml/code/{folder_name}/{estimator_entry_point}"

        if not training_job_completed and path_to_job:
            print(f"Running training job specified in path: {path_to_job}")
            training_job_completed, num_attempts = run_training_job(
                path_to_job, num_attempts
            )

        time.sleep(20)
