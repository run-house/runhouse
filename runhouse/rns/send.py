import pathlib
from typing import Optional, Callable, Dict, Union, List
import os
from pathlib import Path
import glob

from ray import cloudpickle
import ray

from .cluster import Cluster
from .rns_client import RNSClient

class Send:

    def __init__(self,
                 fn: Callable,
                 name=None,
                 working_dir=None,
                 hardware: Union[None, str, Dict[str, int]] = None,
                 reqs: Optional[List[str]] = None,
                 # runtime_env: Union[None, Dict] = None,
                 cluster=None):
        self.name = name

        # Search for working_dir, in the following order:
        # 1. User provided working_dir
        # 2. Environment variable RH_WORKING_DIR
        # 3. If reqs is a path to a requirements.txt, use the parent directory of that
        # 4. Search up the directory tree for a requirements.txt, and use the parent directory if found
        # 5. User's cwd
        self.working_dir = working_dir or os.environ.get('RH_WORKING_DIR', None)
        # Derive working dir from looking up directory tree for requirements.txt
        if self.working_dir is None:
            # Check if reqs is a filepath, and if so, take parent of requirements.txt
            if isinstance(reqs, str) and Path(reqs).exists():
                self.working_dir = Path(reqs).parent
            elif find_requirements_file(os.getcwd()):
                found_reqs_path = Path(find_requirements_file(os.getcwd()))
                self.working_dir = found_reqs_path.parent
            else:
                self.working_dir = os.getcwd()

        config = {}
        if self.name is not None:
            self.send_dir = Path(self.working_dir, "rh/sends", self.name)
            config = RNSClient().load_config_from_name(self.name,
                                                       resource_dir=self.send_dir,
                                                       resource_type='send')

        self.reqs = reqs or config.get('reqs', None)
        # Check if working_dir has requirements.txt, and if so extract reqs
        if Path(self.working_dir, 'requirements.txt').exists():
            # If there are any user-passed reqs, union the requirements.txt reqs with them
            if self.reqs is not None:
                reqstxt_list = Path(self.working_dir, 'requirements.txt').read_text().split('\n')
                passed_reqs_list = []
                if isinstance(self.reqs, str) and Path(self.reqs).exists():
                    passed_reqs_list = Path(self.reqs, 'requirements.txt').read_text().split('\n')
                elif isinstance(self.reqs, list):
                    passed_reqs_list = self.reqs
                else:
                    raise TypeError(f'Send reqs must be either filepath or list, but found {self.reqs}')
                # Ignore version mismatches for now...
                self.reqs = list((set(reqstxt_list) | set(passed_reqs_list)))
            else:
                self.reqs = str(Path(self.working_dir, 'requirements.txt'))

        self.hardware = ({hardware: 1} if isinstance(hardware, str) else hardware) or config.get('hardware', None)
        runtime_env = None

        # For now, default to local if no cluster provided
        cluster = cluster or config.get('cluster', None)
        if cluster is None:
            ray.init(local_mode=True, ignore_reinit_error=True)
        else:
            self.cluster = Cluster(name=cluster, create=True)
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
            # Maybe use Ray's new allow_multiple=True argument to connect to multiple clusters
            # https://docs.ray.io/en/latest/cluster/ray-client.html#connect-to-multiple-ray-clusters-experimental
            if not ray.is_initialized():
                ray.init(address=self.cluster.address,
                         runtime_env=runtime_env,
                         # allow_multiple=True,
                         log_to_driver=False,  # to disable ray workers form logging the output
                         # namespace=self.name,
                         )

        self.fn = fn
        if self.fn is None and config.get('fn', None) is not None:
            self.fn = cloudpickle.loads(config['fn'])

        self.remote_fn = ray.remote(resources=self.hardware)(fn)

        if self.name is not None:
            self.set_name(self.name)

    def __del__(self):
        ray.shutdown()

    def __call__(self, *args, **kwargs):
        res = self.remote_fn.remote(*args, **kwargs)
        return ray.get(res)

    # Name the send, saving it to config stores
    def set_name(self, name):
        sends_path = pathlib.Path(self.working_dir, "rh/sends")
        config = {'name': self.name,
                  # TODO do we need to explicitly pickle by value? (see cloudpickle readme)
                  'fn': cloudpickle.dumps(self.fn),
                  'working_dir': self.working_dir,
                  'hardware': self.hardware,
                  'reqs': self.reqs,
                  'cluster': self.cluster.name,
                  }
        RNSClient.save_to_config(name, resource_dir=sends_path, config=config)

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

def find_requirements_file(dir_path):
    return next(iter(glob.glob(f'{dir_path}/**/requirements.txt', recursive=True)), None)
