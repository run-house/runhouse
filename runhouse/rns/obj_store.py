from pathlib import Path
from typing import Any, Dict, List, Optional

import ray
import ray.cloudpickle as pickle

from runhouse.rns.utils.hardware import _current_cluster


THIS_CLUSTER = _current_cluster("cluster_name")


class ObjStore:
    """Class to handle object storage for Runhouse. Object storage for a cluster is
    stored in the Ray GCS, if available."""

    # Note, if we turn this into a ray actor we could get it by a string on server restart, which
    # would allow us to persist the store across server restarts. Unclear if this is desirable yet.
    # https://docs.ray.io/en/latest/ray-core/package-ref.html#ray-get-actor
    LOGS_DIR = ".rh/logs"
    RH_LOGFILE_PATH = Path.home() / LOGS_DIR

    def __init__(self, cluster_name: Optional[str] = None):
        self.cluster_name = cluster_name or THIS_CLUSTER
        self._obj_store_cache = {}
        self.imported_modules = {}

    @property
    def obj_store_cache(self):
        return self._obj_store_cache

    @obj_store_cache.setter
    def obj_store_cache(self, value: Dict):
        self._obj_store_cache = value

    def put(self, key: str, value: Any):
        obj_ref = ray.put(value)
        self.obj_store_cache[key] = obj_ref

    def put_obj_ref(self, key, obj_ref):
        self.obj_store_cache[key] = obj_ref

    def get(
        self,
        key: str,
        default: Optional[Any] = None,
        timeout: Optional[float] = None,
        resolve: bool = True,
    ):
        obj_ref = self.obj_store_cache.get(key, None)
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
        return list(self.obj_store_cache.keys())

    def delete(self, key: str):
        self.obj_store_cache.pop(key, None)

    def pop(self, key: str, default: Optional[Any] = None):
        return self.obj_store_cache.pop(key, default)

    def clear(self):
        self.obj_store_cache = {}

    def cancel(self, key: str, force: bool = False, recursive: bool = True):
        obj_ref = self.get(key, resolve=False)
        if not obj_ref:
            raise ValueError(f"Object with key {key} not found in object store.")
        if isinstance(obj_ref, list):
            for ref in obj_ref:
                ray.cancel(ref, force=force, recursive=recursive)
        else:
            ray.cancel(obj_ref, force=force, recursive=recursive)

    def get_logfiles(self, key: str, log_type=None):
        # Info on ray logfiles: https://docs.ray.io/en/releases-2.2.0/ray-observability/ray-logging.html#id1
        obj_ref = self.obj_store_cache.get(key, None)
        if obj_ref:
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
        return f"ObjStore({self.obj_store_cache})"

    def __str__(self):
        return f"ObjStore({self.obj_store_cache})"
