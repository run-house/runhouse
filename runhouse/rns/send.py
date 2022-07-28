from typing import Optional, Callable, Dict, Union

import ray

class Send:

    def __init__(self,
                 name=None,):
        self.name = name
        # Check kv store if send exists
        ray.init(local_mode=True)
        pass


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
