import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import ray

from runhouse.servers.utils import LOCALHOST, RH_RAY_PORT

logger = logging.getLogger(__name__)


class ObjStore:
    """A client into shared cluster resources and storage (i.e. servlets). The API largely mirrors the Client API
    (e.g. .get, .put, .call) so they can be interchangeable depending on whether the user is calling a remote or
    local cluster.

    Every Runhouse process has its own ObjStore, including all Envs and the base HTTPServer process. It serves dual
    purposes - it holds the actual Python dict which serves as the in-memory storage for the process, and also acts
    as an intermediary to other Runhouse processes on the cluster (by holding handles to the EnvServlet and
    ClusterServlets actors). This gives us a centralized interface into common resources like Auth, current cluster
    config, looking up the env for a given key, and other Envs' object stores.
    """

    LOGS_DIR = ".rh/logs"
    RH_LOGFILE_PATH = Path.home() / LOGS_DIR

    def __init__(self):
        self.servlet_name = None
        self.cluster_servlet = None
        self._cluster_config_cache = None
        self._kv_store = None
        self.imported_modules = {}
        self.installed_envs = {}
        self._env_servlets_cache = {}

    def cluster_config(self):
        # We cache this because it might require frequent internode communication, even though the config
        # is updated very infrequently
        if self._cluster_config_cache is None:
            self._cluster_config_cache = ray.get(self.cluster_servlet.cluster_config.remote())
        return self._cluster_config_cache

    def set_cluster_config(self, config: Dict):
        self._cluster_config_cache = config
        ray.get(self.cluster_servlet.set_cluster_config.remote(config))

    def set_name(self, servlet_name: str):
        try:
            # if ray.is_initialized():
            #     self.cluster_servlet = ray.get_actor(
            #         "cluster_servlet", namespace="runhouse",
            #     )
            # else:
            from runhouse.servers.cluster_servlet import ClusterServlet

            ray.init(ignore_reinit_error=True,
                     # address=f"ray://{LOCALHOST}:{RH_RAY_PORT}",
                     logging_level=logging.ERROR,
                     namespace="runhouse")
            self.cluster_servlet = ray.remote(ClusterServlet).options(
                name="cluster_servlet",
                get_if_exists=True,
                lifetime="detached",
                namespace="runhouse",
            ).remote()
        except ConnectionError:
            # If ray.init fails, we're not on a cluster, so we don't need to do anything
            pass

        # This needs to be in a separate method so the HTTPServer actually
        # initializes the obj_store, and it doesn't get created and destroyed when
        # nginx runs http_server.py as a module.
        from runhouse.resources.kvstores import Kvstore

        self.servlet_name = servlet_name or "base"
        num_gpus = ray.cluster_resources().get("GPU", 0)
        cuda_visible_devices = list(range(int(num_gpus)))
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, cuda_visible_devices))
        self._kv_store = Kvstore()
        self._kv_store.system = None  # sometimes this gets set to _current_cluster, which can only create problems

    @staticmethod
    def call_actor_method(store, method, *args, **kwargs):
        if store is None:
            raise ValueError(
                "Object store not initialized, may be running inside process without a servlet."
            )
        if isinstance(store, ray.actor.ActorHandle):
            return ray.get(getattr(store, method).remote(*args, **kwargs))
        else:
            return getattr(store, method)(*args, **kwargs)

    #### Auth Methods ####

    def resource_access_level(self, token_hash: str, resource_uri: str):
        return ray.get(
            self.cluster_servlet.lookup_access_level.remote(token_hash, resource_uri)
        )

    def user_resources(self, token_hash: str):
        return ray.get(self.cluster_servlet.user_resources.remote(token_hash))

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

    def clear_auth_cache(self, token_hash: str = None):
        ray.get(self.cluster_servlet.clear_auth_cache.remote(token_hash))

    def add_user(self, token, refresh_cache=True):
        ray.get(self.cluster_servlet.add_user.remote(token, refresh_cache))

    #### Env-Key Lookup Methods ####

    def keys(self, return_envs=False):
        # Return keys across the cluster, not only in this process
        return self.call_actor_method(
            self.cluster_servlet, "keys", return_envs=return_envs
        )

    def list_envs(self, return_keys=False):
        # TODO allow richer state, env configs, key types, etc.
        contents = self.keys(return_envs=True)
        envs = set(contents.values())
        if not return_keys:
            return envs
        # Return a dictionary with envs as keys and the list of keys in the given env as values
        return {env: [key for key, env in contents.items() if env == env] for env in set(envs)}

    def get_env(self, key):
        return self.call_actor_method(self.cluster_servlet, "get_env", key, None)

    def put_env(self, key, value):
        return self.call_actor_method(self.cluster_servlet, "put_env", key, value)

    def put(self, key: str, value: Any, env=None):
        # First check if it's in the Python kv store
        if env and not self.servlet_name == env:
            servlet = self.get_env_servlet(env)
            if servlet is not None:
                if isinstance(servlet, ray.actor.ActorHandle):
                    ray.get(servlet.put.remote(key, value, _intra_cluster=True))
                else:
                    servlet.put(key, value, _intra_cluster=True)

        self.call_actor_method(self._kv_store, "put", key, value)
        self.put_env(key, self.servlet_name)

    def put_obj_ref(self, key, obj_ref):
        # Need to wrap the obj_ref in a dict so ray doesn't dereference it
        # FYI: https://docs.ray.io/en/latest/ray-core/objects.html#closure-capture-of-objects
        self.call_actor_method(self._kv_store, "put", key + "_ref", [obj_ref])
        self.put_env(key, self.servlet_name)

    def rename(self, old_key, new_key, default=None):
        # We also need to rename the resource itself
        obj = self.get(old_key, default=default)
        if obj is not None and hasattr(obj, "rns_address"):
            # Note - we set the obj.name here so the new_key is correctly turned into an rns_address, whether its
            # a full address or just a name. Then, the new_key is set to just the name so its store properly in the
            # kv store.
            obj.name = new_key  # new_key can be an rns_address! e.g. if called by Module.rename
            new_key = obj.name  # new_key is now just the name
        # By passing default, we don't throw an error if the key is not found
        self.call_actor_method(self._kv_store, "rename_key", old_key, new_key, default)
        self.call_actor_method(self.cluster_servlet, "rename_key", old_key, new_key, default)

    def get_obj_ref(self, key):
        return self.call_actor_method(self._kv_store, "get", key + "_ref", [None])[0]

    def get_env_servlet(self, env_name, create=False, runtime_env=None):
        if env_name in self._env_servlets_cache:
            return self._env_servlets_cache[env_name]

        actor = ray.get_actor(env_name, namespace="runhouse")
        if actor is not None:
            self._env_servlets_cache[env_name] = actor
            return actor

        if create:
            from runhouse.servers.env_servlet import EnvServlet

            new_env = (
                ray.remote(EnvServlet)
                .options(
                    name=env_name,
                    get_if_exists=True,
                    runtime_env=runtime_env,
                    lifetime="detached",
                    namespace="runhouse",
                    max_concurrency=1000,
                )
                .remote(env_name=env_name)
            )
            self._env_servlets_cache[env_name] = new_env
            return new_env

        else:
            raise Exception(
                f"Environment {env_name} does not exist. Please send it to the cluster first."
            )

    def get(
        self,
        key: str,
        default: Optional[Any] = None,
        check_other_envs: bool = True,
    ):
        # TODO change this to look up which env the object lives in by default, with an opt out
        # First check if it's in the Python kv store
        try:
            val = self.call_actor_method(self._kv_store, "get", key, KeyError)
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
        self.call_actor_method(self.cluster_servlet, "pop_env", key, default)

    def delete(self, key: Union[str, List[str]]):
        if isinstance(key, str):
            key = [key]
        for k in key:
            self.pop(k, None)
            self.pop_env(k, None)

    def pop(self, key: str, default: Optional[Any] = None):
        res = self.call_actor_method(self._kv_store, "pop", key, default)
        if res:
            self.pop_env(key, None)
        return res

    def clear_env(self):
        self.call_actor_method(self.cluster_servlet, "clear_env")

    def clear(self):
        self.call_actor_method(self._kv_store, "clear")
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
        return self.call_actor_method(self.cluster_servlet, "contains", key)

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
        return f"ObjStore({self.call_actor_method(self._kv_store, '__repr__')})"

    def __str__(self):
        return f"ObjStore({self.call_actor_method(self._kv_store, '__str__')})"
