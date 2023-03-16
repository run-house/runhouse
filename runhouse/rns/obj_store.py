from pathlib import Path
from typing import Any, Dict, List, Optional

import ray

import yaml


def _current_cluster(key="name"):
    """Retrive key value from the current cluster config.
    If key is "config", returns entire config."""
    if Path("~/.rh/cluster_config.yaml").expanduser().exists():
        with open(Path("~/.rh/cluster_config.yaml").expanduser()) as f:
            cluster_config = yaml.safe_load(f)
        if key == "config":
            return cluster_config
        elif key == "cluster_name":
            return cluster_config["name"].rsplit("/", 1)[-1]
        return cluster_config[key]
    else:
        return None


THIS_CLUSTER = _current_cluster("cluster_name")


class ObjStore:
    """Class to handle object storage for Runhouse. Object storage for a cluster is
    stored in the Ray GCS, if available."""

    # Note, if we turn this into a ray actor we could get it by a string on server restart, which
    # would allow us to persist the store across server restarts. Unclear if this is desirable yet.
    # https://docs.ray.io/en/latest/ray-core/package-ref.html#ray-get-actor

    RH_LOGFILE_PATH = Path.home() / ".rh/logs"

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
        self, key: str, default: Optional[Any] = None, timeout: Optional[float] = None
    ):
        obj_ref = self.obj_store_cache.get(key, None)
        if obj_ref:
            return ray.get(obj_ref, timeout=timeout)
        else:
            return default

    def get_obj_refs_list(self, keys: List):
        return [
            self.obj_store_cache.get(key, key) if isinstance(key, str) else key
            for key in keys
        ]

    def get_obj_refs_dict(self, d: Dict):
        return {
            k: self.obj_store_cache.get(v, v) if isinstance(v, str) else v
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
        obj_ref = self.obj_store_cache.get(key, None)
        if obj_ref:
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
