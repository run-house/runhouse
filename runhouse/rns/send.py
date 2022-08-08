import pathlib
from typing import Optional, Callable, Dict, Union, List
import os
from pathlib import Path
import typer
import json
from ray import cloudpickle
import ray
from ray.dashboard.modules.job.sdk import JobSubmissionClient

from .cluster import Cluster
from .rns_client import RNSClient
from ..utils.utils import random_string_generator


class Send:
    RESOURCE_TYPE = 'send'
    SENDS_DIR = "rh/sends"

    def __init__(self,
                 fn: Optional[Callable] = None,
                 hardware: Union[None, str, Dict[str, int]] = None,
                 name=None,
                 working_dir=None,
                 reqs: Optional[List[str]] = None,
                 # runtime_env: Union[None, Dict] = None,
                 cluster=None):
        # Load the client needed to interact with redis
        self.rns_client = RNSClient()
        self.working_dir = self.get_working_dir(working_dir, reqs)

        self.name = name
        if self.name is None:
            # generate a random one if the user didn't provide it
            self.name = random_string_generator()
            typer.echo(f'Created Send with name {self.name}')

        self.send_dir = Path(self.working_dir, self.SENDS_DIR, self.name)

        self.config: dict = self.rns_client.load_config_from_name(self.name, resource_dir=self.send_dir,
                                                                  resource_type=self.RESOURCE_TYPE)
        self.get_reqs(reqs)

        self.hardware = ({hardware: 1} if isinstance(hardware, str) else hardware) or self.config.get('hardware', None)

        # For now, default to local if no cluster provided
        self.create_cluster_for_send(cluster)

        self.fn = fn
        if self.fn is None and self.config.get('fn') is not None:
            self.fn = cloudpickle.loads(bytes(self.config['fn']))

        if self.fn is not None:
            self.remote_fn = ray.remote(resources=self.hardware)(fn)

        self.set_name(self.name)

    def __del__(self):
        ray.shutdown()

    def __call__(self, *args, **kwargs):
        assert self.fn is not None, f"No fn specified for send {self.name}"
        res = self.remote_fn.remote(*args, **kwargs)
        return ray.get(res)

    @property
    def formatted_reqs(self):
        """For storing in redis config cannot handle a list"""
        # if we were given reqs as a list (ex: ['torch'])
        return json.dumps(self.reqs) if isinstance(self.reqs, list) else self.reqs

    def get_working_dir(self, working_dir, reqs):
        # Search for working_dir, in the following order:
        # 1. User provided working_dir
        # 2. Environment variable RH_WORKING_DIR
        # 3. If reqs is a path to a requirements.txt, use the parent directory of that
        # 4. Search up the directory tree for a requirements.txt, and use the parent directory if found
        # 5. User's cwd
        work_dir = working_dir or os.environ.get('RH_WORKING_DIR', None)
        if work_dir is not None:
            return work_dir

        if isinstance(reqs, str) and Path(reqs).exists():
            return str(Path(reqs).parent)

        # Derive working dir from looking up directory tree for requirements.txt
        # Check if reqs is a filepath, and if so, take parent of requirements.txt
        reqs_or_rh_dir = self.find_reqtxt_or_rh(os.getcwd())
        if reqs_or_rh_dir is not None:
            return reqs_or_rh_dir
        else:
            return os.getcwd()

    @staticmethod
    def find_reqtxt_or_rh(dir_path):
        if Path(dir_path) == Path.home():
            return None
        if Path(dir_path, 'requirements.txt').exists() or Path(dir_path, 'rh').exists():
            return str(dir_path)
        else:
            return Send.find_reqtxt_or_rh(Path(dir_path).parent)

    def get_reqs(self, reqs):
        self.reqs = reqs or self.config.get('reqs', None)
        # Check if working_dir has requirements.txt, and if so extract reqs
        if Path(self.working_dir, 'requirements.txt').exists():
            # If there are any user-passed reqs, union the requirements.txt reqs with them
            if self.reqs is not None:
                reqstxt_list = Path(self.working_dir, 'requirements.txt').read_text().split('\n')
                if (isinstance(self.reqs, str) or isinstance(self.reqs, Path)) and Path(self.reqs).exists():
                    passed_reqs_list = Path(self.reqs, 'requirements.txt').read_text().split('\n')
                elif isinstance(self.reqs, list):
                    passed_reqs_list = self.reqs
                else:
                    raise TypeError(f'Send reqs must be either filepath or list, but found {self.reqs}')
                # Ignore version mismatches for now...
                self.reqs = list((set(reqstxt_list) | set(passed_reqs_list)))
            else:
                self.reqs = str(Path(self.working_dir, 'requirements.txt'))

    def create_cluster_for_send(self, cluster):
        cluster = cluster or self.config.get('cluster', None)
        if cluster is None:
            ray.init(local_mode=True, ignore_reinit_error=True)
        else:
            self.cluster = Cluster(name=cluster, create=True, rns_client=self.rns_client)
            os.environ['RAY_IGNORE_VERSION_MISMATCH'] = 'True'
            # TODO merge with user-supplied runtime_env
            # TODO see if we can fix the python mismatch here via the 'conda' argument
            runtime_env = {"working_dir": self.working_dir,
                           "pip": self.reqs,
                           'env_vars': dict(os.environ),
                           'excludes': ['*.log', '*.tar', '*.tar.gz', '.env', 'venv', '.idea', '.DS_Store',
                                        '__pycache__',
                                        '*.whl']}
            # TODO for now, different sends on the same cluster have to share these, we should fix soon
            # Also, maybe we'll need to isolate the connection to allow multiple:
            # https://docs.ray.io/en/latest/cluster/ray-client.html#connect-to-multiple-ray-clusters-experimental
            if not ray.is_initialized():
                # TODO figure out proper addressing if we're already inside of ray cluster
                ray.init(address=self.cluster.address,
                         runtime_env=runtime_env,
                         # allow_multiple=True,
                         log_to_driver=False,  # to disable ray workers form logging the output
                         # namespace=self.name,
                         )

    # Name the send, saving it to config stores
    def set_name(self, name):
        sends_path = pathlib.Path(self.working_dir, self.SENDS_DIR)
        config = {'name': self.name,
                  'working_dir': self.working_dir,
                  'hardware': self.hardware,
                  'reqs': self.formatted_reqs,
                  'cluster': self.cluster.name,
                  # TODO do we need to explicitly pickle by value? (see cloudpickle readme)
                  'fn': str(cloudpickle.dumps(self.fn)),
                  }
        self.rns_client.save_config_for_name(name=name, config=config, resource_dir=sends_path,
                                             resource_type=self.RESOURCE_TYPE)

    def run(self, run_cmd):
        client = JobSubmissionClient(f"http://{self.cluster.cluster_ip}:8265")
        job_id = client.submit_job(
            entrypoint=run_cmd,
            runtime_env={
                "working_dir": self.working_dir,
                "pip": self.find_reqtxt_or_rh(self.working_dir)
            }
        )

    def map(self, replicas=1):
        # TODO
        # https://docs.ray.io/en/latest/ray-core/tasks/patterns/map-reduce.html
        # return ray.get([map.remote(i, map_func) for i in replicas])
        pass
