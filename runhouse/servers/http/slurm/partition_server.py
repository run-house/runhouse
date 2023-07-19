import json
import logging
import subprocess
import time
from pathlib import PosixPath
from typing import List, Union

from runhouse.rh_config import obj_store

from runhouse.servers.http.http_utils import b64_unpickle

logger = logging.getLogger(__name__)


class SlurmPartitionServer:
    """Reads requests from the queue in the Slurm cluster's shared filesystem, and executes the latest one.
    The queue is represented by the job directories created inside the ~/.rh/jobs folder of the cluster.
    """

    CHECK_QUEUE_INTERVAL = 2  # seconds
    QUEUE_PATH = obj_store.RH_LOGFILE_PATH
    REQUEST_FILE = "request.json"

    @classmethod
    def path_to_job_request(cls, job_folder):
        return job_folder / cls.REQUEST_FILE

    @classmethod
    def queue(cls):
        all_jobs = []
        for job_folder in cls.QUEUE_PATH.iterdir():
            request_file = cls.path_to_job_request(job_folder)
            if not request_file.exists():
                # Job with same name may have been submitted again, or other folders may exist without the request file
                continue
            all_jobs.append(job_folder)
        return all_jobs

    @classmethod
    def pop_queue(cls) -> Union["rh.Folder", None]:
        """Get the latest job from the queue. Returns a folder object for the latest job if one exists,
        otherwise returns None."""
        from runhouse import folder

        jobs_in_queue: list = cls.queue()
        if not jobs_in_queue:
            return

        logger.info(f"Jobs in queue: {len(jobs_in_queue)}")

        job_to_run = cls.get_latest_job(jobs_in_queue)

        if job_to_run is None:
            return None

        # Return a runhouse folder object
        return folder(name=job_to_run.name, path=str(job_to_run))

    @classmethod
    def get_latest_job(cls, job_queue: List[PosixPath]) -> Union[PosixPath, None]:
        latest_modified_time = None
        latest_job = None

        for job in job_queue:
            request_file = cls.path_to_job_request(job)
            modified_time = request_file.stat().st_mtime
            if latest_modified_time is None or modified_time > latest_modified_time:
                latest_modified_time = modified_time
                latest_job = job

        return latest_job

    # TODO [JL] WIP - need to work out the way the partition server runs functions or executes commands
    @classmethod
    def run(cls):
        while True:
            try:
                current_job: "rh.Folder" = cls.pop_queue()
                if current_job is None:
                    time.sleep(cls.CHECK_QUEUE_INTERVAL)
                    continue

                logger.info(f"Running job: {current_job.name}")

                # Read the pickled message data from the request sent to the Jump Server that is saved in
                # the job's dedicated folder
                with open(f"{current_job.path}/{cls.REQUEST_FILE}", "r") as f:
                    request_data = json.load(f)

                logger.info(f"Request data: {request_data}")
                (
                    name,
                    partition,
                    fn_obj,
                    commands,
                    env,
                    mail_type,
                    mail_user,
                    args,
                    kwargs,
                ) = b64_unpickle(request_data)

                # TODO [JL] support other params here? (ex: ntasks, nodes, etc.)
                command = f"""srun --partition={partition} --job-name={name} --pty bash -c {commands}"""

                if mail_user and mail_type:
                    # Enable slurm email notifications if mail_user and mail_type are provided
                    mail_options = f"--mail-type={mail_type} --mail-user={mail_user}"
                    command.replace("srun", f"srun {mail_options}", 1)

                # Run the slurm command to be executed on the partition server
                result = subprocess.run(commands, capture_output=True, text=True)

                # Access the captured output saved on the jump server
                stdout = result.stdout
                stderr = result.stderr

                # Save down the stdout and stderr to the job's folder on the jump server
                current_job.put({f"{name}.out": stdout}, overwrite=True)
                current_job.put({f"{name}.err": stderr}, overwrite=True)

                logger.info(
                    f"Saved stdout and stderr for job {name} to folder: {str(current_job.path)}"
                )

                # remove the job's request file from its folder once its been executed
                current_job.rm(contents=cls.REQUEST_FILE)

                # TODO [JL] save the completed job to a different queue for completed jobs? (ex: ~/.rh/completed_jobs)

            except Exception as e:
                logger.exception(e)
                time.sleep(cls.CHECK_QUEUE_INTERVAL)
                continue


if __name__ == "__main__":
    SlurmPartitionServer.run()
