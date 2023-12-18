import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import ray

logger = logging.getLogger(__name__)


class ObjStore:
    """Class to handle object storage for Runhouse. Object storage for a cluster is
    stored in the Ray GCS, if available."""

    # Note, if we turn this into a ray actor we could get it by a string on server restart, which
    # would allow us to persist the store across server restarts. Unclear if this is desirable yet.
    # https://docs.ray.io/en/latest/ray-core/package-ref.html#ray-get-actor
    LOGS_DIR = ".rh/logs"
    RH_LOGFILE_PATH = Path.home() / LOGS_DIR

    def __init__(self):
        self.servlet_name = None
        self._kv_store = None
        self._env_for_key = None
        self.imported_modules = {}
        self.installed_envs = {}
        self._auth_cache = None

    def set_name(self, servlet_name: str):
        # This needs to be in a separate method so the HTTPServer actually
        # initalizes the obj_store, and it doesn't get created and destroyed when
        # nginx runs http_server.py as a module.
        from runhouse.resources.kvstores import Kvstore
        from runhouse.servers.http.auth import AuthCache

        self.servlet_name = servlet_name or "base"
        num_gpus = ray.cluster_resources().get("GPU", 0)
        cuda_visible_devices = list(range(int(num_gpus)))
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, cuda_visible_devices))
        self._kv_store = Kvstore()
        self._kv_store.system = None  # sometimes this gets set to _current_cluster, which can only create problems
        self._env_for_key = (
            ray.remote(Kvstore)
            .options(
                name="env_for_key",
                get_if_exists=True,
                lifetime="detached",
                namespace="runhouse",
            )
            .remote(
                system="here"
            )  # Same here, we don't want to use the _current_cluster system
        )
        self._auth_cache = (
            ray.remote(AuthCache)
            .options(
                name="auth_cache",
                get_if_exists=True,
                lifetime="detached",
                namespace="runhouse",
            )
            .remote()
        )

    @staticmethod
    def call_kv_method(store, method, *args, **kwargs):
        if store is None:
            raise ValueError(
                "Object store not initialized, may be running inside process without a servlet."
            )
        if isinstance(store, ray.actor.ActorHandle):
            return ray.get(getattr(store, method).remote(*args, **kwargs))
        else:
            return getattr(store, method)(*args, **kwargs)

    def resource_access_level(self, token_hash: str, resource_uri: str):
        return ray.get(
            self._auth_cache.lookup_access_level.remote(token_hash, resource_uri)
        )

    def user_resources(self, token_hash: str):
        return ray.get(self._auth_cache.get_user_resources.remote(token_hash))

    def has_resource_access(self, token_hash: str, resource_uri=None) -> bool:
        """Checks whether user has read or write access to a given module saved on the cluster."""
        from runhouse.rns.utils.api import ResourceAccess
        from runhouse.servers.http.http_utils import load_current_cluster

        if token_hash is None:
            # If no token is provided assume no access
            return False

        cluster_uri = load_current_cluster()
        cluster_access = self.resource_access_level(token_hash, cluster_uri)
        if cluster_access == ResourceAccess.WRITE:
            # if user has write access to cluster will have access to all resources
            return True

        if resource_uri is None and cluster_access not in [
            ResourceAccess.WRITE,
            ResourceAccess.READ,
        ]:
            # If module does not have a name, must have access to the cluster
            return False

        resource_access_level = self.resource_access_level(token_hash, resource_uri)
        if resource_access_level not in [ResourceAccess.WRITE, ResourceAccess.READ]:
            return False

        return True

    def keys(self, return_envs=False):
        # Return keys across the cluster, not only in this process
        return self.call_kv_method(
            self._env_for_key, "items" if return_envs else "keys"
        )

    def get_env(self, key):
        return self.call_kv_method(self._env_for_key, "get", key, None)

    def put_env(self, key, value):
        return self.call_kv_method(self._env_for_key, "put", key, value)

    def put(self, key: str, value: Any, env=None):
        # First check if it's in the Python kv store
        if env and not self.servlet_name == env:
            servlet = self.get_env_servlet(env)
            if servlet is not None:
                if isinstance(servlet, ray.actor.ActorHandle):
                    ray.get(servlet.put.remote(key, value, _intra_cluster=True))
                else:
                    servlet.put(key, value, _intra_cluster=True)

        self.call_kv_method(self._kv_store, "put", key, value)
        self.put_env(key, self.servlet_name)

    def put_obj_ref(self, key, obj_ref):
        # Need to wrap the obj_ref in a dict so ray doesn't dereference it
        # FYI: https://docs.ray.io/en/latest/ray-core/objects.html#closure-capture-of-objects
        self.call_kv_method(self._kv_store, "put", key + "_ref", [obj_ref])
        self.put_env(key, self.servlet_name)

    def rename(self, old_key, new_key, default=None):
        # By passing default, we don't throw an error if the key is not found
        self.call_kv_method(self._kv_store, "rename_key", old_key, new_key, default)
        self.call_kv_method(self._env_for_key, "rename_key", old_key, new_key, default)

    def get_obj_ref(self, key):
        return self.call_kv_method(self._kv_store, "get", key + "_ref", [None])[0]

    @staticmethod
    def get_env_servlet(env_name):
        from runhouse.globals import env_servlets

        if env_name in env_servlets.keys():
            return env_servlets[env_name]

        actor = ray.get_actor(env_name, namespace="runhouse")
        if actor is not None:
            env_servlets[env_name] = actor
            return actor
        return None

    def get(
        self,
        key: str,
        default: Optional[Any] = None,
        check_other_envs: bool = True,
    ):
        # TODO change this to look up which env the object lives in by default, with an opt out
        # First check if it's in the Python kv store
        try:
            val = self.call_kv_method(self._kv_store, "get", key, KeyError)
            return val
        except KeyError as e:
            key_err = e

        if not check_other_envs:
            if default == KeyError:
                raise key_err
            return default

        # If not, check if it's in another env's servlet
        servlet_name = self.get_env(key)
        if servlet_name is None:
            if default == KeyError:
                raise key_err
            return default

        logger.info(f"Getting {key} from servlet {servlet_name}")
        servlet = self.get_env_servlet(servlet_name)
        if servlet is None:
            if default == KeyError:
                raise key_err
            return default

        try:
            if isinstance(servlet, ray.actor.ActorHandle):
                return ray.get(servlet.get.remote(key, _intra_cluster=True))
            else:
                return servlet.get(key, _intra_cluster=True, timeout=None)
        except KeyError as e:
            if default == KeyError:
                raise e

        return default

    def get_list(self, keys: List[str], default: Optional[Any] = None):
        return [self.get(key, default=default or key) for key in keys]

    def get_obj_refs_list(self, keys: List):
        return [
            self.get(key, default=key) if isinstance(key, str) else key for key in keys
        ]

    def get_obj_refs_dict(self, d: Dict):
        return {
            k: self.get(v, default=v) if isinstance(v, str) else v for k, v in d.items()
        }

    def pop_env(self, key: str, default: Optional[Any] = None):
        self.call_kv_method(self._env_for_key, "pop", key, default)

    def delete(self, key: Union[str, List[str]]):
        if isinstance(key, str):
            key = [key]
        for k in key:
            self.pop(k, None)
            self.pop_env(k, None)

    def pop(self, key: str, default: Optional[Any] = None):
        res = self.call_kv_method(self._kv_store, "pop", key, default)
        if res:
            self.pop_env(key, None)
        return res

    def clear_env(self):
        self.call_kv_method(self._env_for_key, "clear")

    def clear(self):
        self.call_kv_method(self._kv_store, "clear")
        self.clear_env()

    def cancel(self, key: str, force: bool = False, recursive: bool = True):
        # TODO wire up properly
        obj_ref = self.get_obj_ref(key)
        if not obj_ref:
            raise ValueError(f"Object with key {key} not found in object store.")
        else:
            ray.cancel(obj_ref, force=force, recursive=recursive)

    def cancel_all(self, force: bool = False, recursive: bool = True):
        for key in self.keys():
            self.cancel(key, force=force, recursive=recursive)

    def contains(self, key: str):
        return self.call_kv_method(self._env_for_key, "contains", key)

    def get_logfiles(self, key: str, log_type=None):
        # TODO remove
        # Info on ray logfiles: https://docs.ray.io/en/releases-2.2.0/ray-observability/ray-logging.html#id1
        if self.contains(key):
            # Logs are like: `.rh/logs/key.[out|err]`
            key_logs_path = Path(self.RH_LOGFILE_PATH) / key
            glob_pattern = (
                "*.out"
                if log_type == "stdout"
                else "*.err"
                if log_type == "stderr"
                else "*.[oe][ur][tr]"
            )
            return [str(f.absolute()) for f in key_logs_path.glob(glob_pattern)]
        else:
            return None

    def __repr__(self):
        return f"ObjStore({self.call_kv_method(self._kv_store, '__repr__')})"

    def __str__(self):
        return f"ObjStore({self.call_kv_method(self._kv_store, '__str__')})"
