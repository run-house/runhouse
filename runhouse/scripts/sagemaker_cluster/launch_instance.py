"""Script used to keep the SageMaker cluster up, pending the autostop time provided in the cluster's config."""
import json
import os
import subprocess
import time
import warnings
from pathlib import Path
from typing import Dict

DEFAULT_AUTOSTOP = -1
MAIN_DIR = "/opt/ml/code"
OUT_FILE = "sm_cluster.out"

# ---------Configure SSH Helper----------
# https://github.com/aws-samples/sagemaker-ssh-helper#step-3-modify-your-training-script
import sagemaker_ssh_helper

sagemaker_ssh_helper.setup_and_start_ssh()


def run_training_job(path_to_job: str, num_attempts: int):
    job_succeeded = False

    try:
        # Execute the script as a separate process and capture stdout and stderr
        completed_process = subprocess.run(
            ["python", path_to_job],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # Redirect stderr to stdout
            text=True,
        )

        # Access combined stdout and stderr
        combined_output = completed_process.stdout

        if combined_output:
            print(combined_output)
            with open(f"{MAIN_DIR}/{OUT_FILE}", "w") as f:
                f.write(combined_output)

        job_succeeded = True

    except subprocess.CalledProcessError as e:
        warnings.warn(
            f"({e.returncode}) Error executing training script "
            f"(already made {num_attempts} attempts): {e.stderr}"
        )
        num_attempts += 1

    except Exception as e:
        num_attempts += 1
        warnings.warn(
            f"Error executing training script (already made {num_attempts} attempts): {e}"
        )

    finally:
        return job_succeeded, num_attempts


def read_cluster_config() -> Dict:
    """Read the autostop from the cluster's config - this will get populated when the cluster is created,
    or via the autostop APIs (e.g. `pause_autostop` or `keep_warm`)"""
    try:
        # Note: Runhouse has not yet been installed at this stage on the cluster,
        # so we can't import CLUSTER_CONFIG_PATH, we just need to hardcode it.
        with open(os.path.expanduser("~/.rh/cluster_config.json"), "r") as f:
            cluster_config = json.load(f)
    except FileNotFoundError:
        cluster_config = {}

    return cluster_config


if __name__ == "__main__":
    print("Launching instance from script")
    server_log = Path("~/.rh/server.log").expanduser()
    server_log.parent.mkdir(parents=True, exist_ok=True)
    server_log.touch()

    last_active = time.time()
    last_autostop_value = None
    training_job_completed = False
    path_to_job = None
    num_attempts = 0

    while True:
        last_active = server_log.stat().st_mtime
        config = read_cluster_config()

        autostop = int(config.get("autostop_mins", DEFAULT_AUTOSTOP))

        if autostop != -1:
            time_to_autostop = last_active + (autostop * 60)
            current_time = time.time()

            if current_time >= time_to_autostop:
                print("Autostop time reached, stopping instance")
                break

            # Reset launch time if autostop was updated
            if last_autostop_value is not None and autostop != last_autostop_value:
                print(f"Resetting autostop from {last_autostop_value} to {autostop}")
                last_active = current_time

            last_autostop_value = autostop

        estimator_entry_point = config.get("estimator_entry_point")
        if estimator_entry_point:
            # Update the path to the custom estimator job which was rsynced to the cluster
            # and now lives in path: /opt/ml/code
            path_to_job = f"{MAIN_DIR}/{estimator_entry_point}"

        if not training_job_completed and path_to_job:
            print(f"Running training job specified in path: {path_to_job}")
            training_job_completed, num_attempts = run_training_job(
                path_to_job, num_attempts
            )

        time.sleep(20)
