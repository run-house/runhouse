import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import ray
import ray.cloudpickle as pickle

from runhouse.rns.utils.hardware import _current_cluster


@ray.remote
class ObjStoreActor:
    """Ray actor to handle object storage for Runhouse. Wrapping this in an actor allows us to access it across
    Ray processes and nodes, and even keep some things pinned to Python memory."""

    def __init__(self):
        num_gpus = ray.cluster_resources().get("GPU", 0)
        cuda_visible_devices = list(range(int(num_gpus)))
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, cuda_visible_devices))
        self._kv = {}

    def put(self, key: str, value: Any):
        self._kv[key] = value

    def get(self, key: str, default=None):
        return self._kv.get(key, default)

    def pop(self, key: str, *args):
        # We accept *args here to match the signature of dict.pop (throw an error if key is not found,
        # unless another arg is provided as a default)
        return self._kv.pop(key, *args)

    def keys(self):
        return list(self._kv.keys())

    def values(self):
        return list(self._kv.values())

    def items(self):
        return list(self._kv.items())

    def clear(self):
        self._kv = {}

    def rename(self, old_key, new_key, *args):
        # We accept *args here to match the signature of dict.pop (throw an error if key is not found,
        # unless another arg is provided as a default)
        self._kv[new_key] = self._kv.pop(old_key, *args)

    def __len__(self):
        return len(self._kv)

    def __contains__(self, key: str):
        return key in self._kv

    def __getitem__(self, key: str):
        return self._kv[key]

    def __setitem__(self, key: str, value: Any):
        self._kv[key] = value

    def __delitem__(self, key: str):
        del self._kv[key]

    def __repr__(self):
        return repr(self._kv)


class ObjStore:
    """Class to handle object storage for Runhouse. Object storage for a cluster is
    stored in the Ray GCS, if available."""

    # Note, if we turn this into a ray actor we could get it by a string on server restart, which
    # would allow us to persist the store across server restarts. Unclear if this is desirable yet.
    # https://docs.ray.io/en/latest/ray-core/package-ref.html#ray-get-actor
    LOGS_DIR = ".rh/logs"
    RH_LOGFILE_PATH = Path.home() / LOGS_DIR

    def __init__(self, cluster_name: Optional[str] = None):
        if not ray.is_initialized():
            ray.init(
                ignore_reinit_error=True,
                namespace="runhouse_server",
            )

        self.cluster_name = cluster_name or _current_cluster("cluster_name")
        num_gpus = ray.cluster_resources().get("GPU", 0)
        cuda_visible_devices = list(range(int(num_gpus)))
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, cuda_visible_devices))
        self._kv_actor = ObjStoreActor.options(
            name="obj_store_kv",
            get_if_exists=True,
            lifetime="detached",
            namespace="runhouse_server",
        ).remote()
        self.imported_modules = {}

    def put(self, key: str, value: Any):
        if not isinstance(value, ray.ObjectRef):
            ray.get(self._kv_actor.put.remote(key, value))
            value = ray.put(value)
        self.put_obj_ref(key, value)

    def put_obj_ref(self, key, obj_ref):
        # Need to wrap the obj_ref in a dict so ray doesn't dereference it
        # FYI: https://docs.ray.io/en/latest/ray-core/objects.html#closure-capture-of-objects
        ray.get(self._kv_actor.put.remote(key + "_ref", [obj_ref]))

    def rename(self, old_key, new_key, default=None):
        # By passing default, we don't throw an error if the key is not found
        ray.get(self._kv_actor.rename.remote(old_key, new_key, default))
        ray.get(
            self._kv_actor.rename.remote(old_key + "_ref", new_key + "_ref", default)
        )

    def get_obj_ref(self, key):
        return ray.get(self._kv_actor.get.remote(key + "_ref", [None]))[0]

    def get(
        self,
        key: str,
        default: Optional[Any] = None,
        timeout: Optional[float] = None,
        resolve: bool = True,
    ):
        # First check if it's in the Python kv store
        val = ray.get(self._kv_actor.get.remote(key))
        if val is not None:
            return val

        # Next check if it's in the Ray Obj store
        obj_ref = self.get_obj_ref(key)

        if not obj_ref:
            return default

        if not resolve:
            return obj_ref

        obj = ray.get(obj_ref, timeout=timeout)
        if isinstance(obj, bytes):
            return pickle.loads(obj)
        else:
            return obj

    def get_obj_refs_list(self, keys: List, resolve=True):
        return [
            self.get(key, default=key, resolve=resolve) if isinstance(key, str) else key
            for key in keys
        ]

    def get_obj_refs_dict(self, d: Dict, resolve=True):
        return {
            k: self.get(v, default=v, resolve=resolve) if isinstance(v, str) else v
            for k, v in d.items()
        }

    def keys(self):
        return list(ray.get(self._kv_actor.keys.remote()))

    def delete(self, key: str):
        ray.get(self._kv_actor.pop.remote(key, None))

    def pop(self, key: str, default: Optional[Any] = None):
        val = ray.get(self._kv_actor.pop.remote(key, default))
        ref = ray.get(self._kv_actor.pop.remote(key + "_ref", [None])[0])
        return val or ref or default

    def clear(self):
        ray.get(self._kv_actor.clear.remote())

    def cancel(self, key: str, force: bool = False, recursive: bool = True):
        obj_ref = self.get_obj_ref(key)
        if not obj_ref:
            raise ValueError(f"Object with key {key} not found in object store.")
        else:
            ray.cancel(obj_ref, force=force, recursive=recursive)

    def contains(self, key: str):
        return ray.get(self._kv_actor.__contains__.remote(key + "_ref")) or ray.get(
            self._kv_actor.__contains__.remote(key)
        )

    def get_logfiles(self, key: str, log_type=None):
        # Info on ray logfiles: https://docs.ray.io/en/releases-2.2.0/ray-observability/ray-logging.html#id1
        if self.contains(key):
            # Logs are like worker-[worker_id]-[job_id]-[pid].[out|err]
            key_logs_path = Path(self.RH_LOGFILE_PATH) / key
            # stdout_files = ray_logs_path.glob(f'worker-*-{obj_ref.job_id().hex()}-*.out')
            suffix = (
                ".out"
                if log_type == "stdout"
                else ".err"
                if log_type == "stderr"
                else ""
            )
            return [str(f.absolute()) for f in key_logs_path.glob(f"worker*{suffix}")]
        else:
            return None

    def __repr__(self):
        return f"ObjStore({ray.get(self._kv_actor.__repr__.remote())})"

    def __str__(self):
        return f"ObjStore({ray.get(self._kv_actor.__repr__.remote())})"
