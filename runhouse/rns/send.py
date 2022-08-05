from typing import Optional, Callable, Dict, Union
import os
from pathlib import Path

import ray

from .cluster import Cluster

class Send:

    def __init__(self,
                 name=None,
                 working_dir=None,
                 reqs=None,
                 hardware=None,
                 cluster=None):
        self.name = name
        self.working_dir = working_dir or os.getcwd()
        self.reqs = reqs
        self.hardware = hardware

        if self.name is not None:
            # Get or create rh, sends, and name dir in the working directory
            Path(self.working_dir, "rh/sends", self.name).mkdir(parents=True, exist_ok=True)

            # TODO get_or_create_send_config
            # TODO Check kv store if send exists, pull down config.

        # For now, default to local
        if cluster is None:
            ray.init(local_mode=True, ignore_reinit_error=True)
        else:
            self.cluster = Cluster(name=cluster, create=True)
            os.environ['RAY_IGNORE_VERSION_MISMATCH'] = 'True'
            ray.init(address=self.cluster.address)
            # runtime_env = {"working_dir": self.working_dir,
            #                "pip": self.reqs,
            #                'env_vars': dict(os.environ),
            #                'excludes': ['*.log', '*.tar', '*.tar.gz', '.env', 'venv', '.idea', '.DS_Store',
            #                             '__pycache__',
            #                             '*.whl']}
            # # use the remote cluster head node's IP address
            # # TODO resolve hardware name to ip
            # ray.init(f'ray://{self.cluster.address}',
            #          # namespace=self.name,
            #          runtime_env=runtime_env,
            #          log_to_driver=False)  # to disable ray workers form logging the output

        if self.name is not None:
            pass
            # TODO Save down updates into local config
            # TODO update config in kv store
    def run(self, run_cmd):
        client = JobSubmissionClient(f"http://{hardware_ip}:8265")
        job_id = client.submit_job(
            entrypoint=run_cmd,
            runtime_env={
                "working_dir": path_to_runnable_file,
                "pip": reqs_file
            }
        )

    def map(self, replicas=1):
        # TODO
        # https://docs.ray.io/en/latest/ray-core/tasks/patterns/map-reduce.html
        return ray.get([map.remote(i, map_func) for i in replicas])


def send(fn: Callable,
         name=None,
         hardware: Union[None, str, Dict[str, int]] = None,
         runtime_env: Union[None, Dict] = None,
         cluster=None
         ):
    new_send = Send(name=name,
                    hardware=hardware,
                    cluster=cluster
                    )
    # Resolve hardware properly through rh API (like just pull down the resources info to pass to ray)
    remote_fn = ray.remote(resources=hardware,
                           runtime_env=runtime_env)(fn)

    def sent_fn(*args, **kwargs):
        res = remote_fn.remote(*args, **kwargs)
        return ray.get(res)
    return sent_fn
