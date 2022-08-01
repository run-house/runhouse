from typing import Optional, Callable, Dict, Union
import os

import ray

class Send:

    def __init__(self,
                 name=None,
                 package_path=None,
                 reqs=None,
                 hardware='local',
                 cluster_yaml_path=None,
                 cluster_ip=None):
        self.name = name
        self.package_path = package_path
        self.reqs = reqs
        self.hardware = hardware
        self.cluster_yaml_path = cluster_yaml_path
        self.cluster_ip = cluster_ip
        # TODO Check kv store if send exists
        if hardware == 'local':
            ray.init(local_mode=True)
        else:
            os.environ['RAY_IGNORE_VERSION_MISMATCH'] = 'True'
            runtime_env = {"working_dir": self.package_path,
                           "pip": self.reqs,
                           'env_vars': dict(os.environ),
                           'excludes': ['*.log', '*.tar', '*.tar.gz', '.env', 'venv', '.idea', '.DS_Store',
                                        '__pycache__',
                                        '*.whl']}
            # use the remote cluster head node's IP address
            # TODO resolve hardware name to ip
            ray.init(f'ray://{self.cluster_ip}:10001',
                     namespace=self.name,
                     runtime_env=runtime_env,
                     log_to_driver=False)  # to disable ray workers form logging the output


def send(fn: Callable,
         hardware: Union[None, str, Dict[str, int]] = None,
         runtime_env: Union[None, Dict] = None
         ):
    # Resolve hardware properly through rh API (like just pull down the resources info to pass to ray)
    remote_fn = ray.remote(resources=hardware,
                           runtime_env=runtime_env)(fn)

    def sent_fn(*args, **kwargs):
        res = remote_fn.remote(*args, **kwargs)
        return ray.get(res)
    return sent_fn
