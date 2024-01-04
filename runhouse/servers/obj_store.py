import logging
import os
from typing import Any, Dict, List, Optional, Set, Union

import ray

logger = logging.getLogger(__name__)


class ObjStoreError(Exception):
    pass


class NoLocalObjStoreError(ObjStoreError):
    def __init__(self):
        super().__init__("No local object store exists; cannot perform operation.")


class ObjStore:
    """Class to handle internal IPC and storage for Runhouse.

    We interact with individual EnvServlets as well as the global ClusterServlet
    via this class.

    The point of this is that this information can
    be accessed by any node in the cluster, as well as any
    process on any node, via this class.

    1. We store the state of the cluster in the ClusterServlet.
    2. We store an auth cache in the ClusterServlet
    3. We interact with a distributed KV store, which is the most in-depth of these use cases.

        The KV store is used to store objects that are shared across the cluster. Each EnvServlet
        will have its own ObjStore initialized with a servlet name. This means it will have a
        local Python dictionary with its values. However, each ObjStore can also access the other env
        servlets' KV stores, so we can get and put values across the cluster.

        We maintain individual KV stores in each EnvServlet's memory so that we can access them in-memory
        if functions within that Servlet make key/value requests.
    """

    def __init__(self):
        self.servlet_name: Optional[str] = None
        self.cluster_servlet: Optional[ray.actor.ActorHandle] = None
        self.imported_modules = {}
        self.installed_envs = {}

        self._kv_store: Dict[Any, Any] = None

    def initialize(
        self, servlet_name: Optional[str] = None, has_local_storage: bool = False
    ):
        # The initialization of the obj_store needs to be in a separate method
        # so the HTTPServer actually initalizes the obj_store,
        # and it doesn't get created and destroyed when
        # nginx runs http_server.py as a module.

        # ClusterServlet essentially functions as a global state/metadata store
        # for all nodes connected to this Ray cluster.
        try:
            from runhouse.servers.cluster_servlet import ClusterServlet

            ray.init(
                ignore_reinit_error=True,
                logging_level=logging.ERROR,
                namespace="runhouse",
            )
            self.cluster_servlet = (
                ray.remote(ClusterServlet)
                .options(
                    name="cluster_servlet",
                    get_if_exists=True,
                    lifetime="detached",
                    namespace="runhouse",
                )
                .remote()
            )

            # Make sure ClusterServlet is fully initialized
            ray.get(self.cluster_servlet.get_cluster_config.remote())

        except ConnectionError:
            # If ray.init fails, we're not on a cluster, so we don't need to do anything
            pass

        # There are 3 operating modes of the KV store:
        # servlet_name is set, has_local_storage is True: This is an EnvServlet with a local KV store.
        # servlet_name is set, has_local_storage is False: This is an ObjStore class that is not an EnvServlet,
        #   but wants to proxy its writes to a running EnvServlet.
        # servlet_name is unset, has_local_storage is False: This is an ObjStore class that by default only looks at
        #   the global KV store and other servlets.
        if not servlet_name and has_local_storage:
            raise ValueError(
                "Must provide a servlet name if the servlet has local storage."
            )

        if (
            servlet_name
            and not has_local_storage
            and not self.get_env_servlet(servlet_name)
        ):
            raise ValueError(
                f"ObjStore wants to proxy writes to {servlet_name}, but there is no servlet with that name running."
            )

        # There can only be one initialized EnvServlet with a given name AND with local storage.
        if has_local_storage and servlet_name:
            if self.is_env_servlet_name_initialized(servlet_name):
                raise ValueError(
                    f"There already exists an EnvServlet with name {servlet_name}."
                )
            else:
                self.mark_env_servlet_name_as_initialized(servlet_name)

        self.servlet_name = servlet_name
        self.has_local_storage = has_local_storage
        if self.has_local_storage:
            self._kv_store = {}

        num_gpus = ray.cluster_resources().get("GPU", 0)
        cuda_visible_devices = list(range(int(num_gpus)))
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, cuda_visible_devices))

    ##############################################
    # Generic helpers
    ##############################################
    @staticmethod
    def call_actor_method(actor: ray.actor.ActorHandle, method: str, *args, **kwargs):
        if actor is None:
            raise ObjStoreError("Attempting to call an actor method on a None actor.")
        return ray.get(getattr(actor, method).remote(*args, **kwargs))

    @staticmethod
    def get_env_servlet(env_name: str):
        from runhouse.globals import env_servlets

        if env_name in env_servlets.keys():
            return env_servlets[env_name]

        actor = ray.get_actor(env_name, namespace="runhouse")
        if actor is not None:
            env_servlets[env_name] = actor
            return actor
        return None

    ##############################################
    # Cluster config state storage methods
    ##############################################
    def get_cluster_config(self):
        # TODO: Potentially add caching here
        if self.cluster_servlet is not None:
            return self.call_actor_method(self.cluster_servlet, "get_cluster_config")
        else:
            return {}

    def set_cluster_config(self, config: Dict[str, Any]):
        return self.call_actor_method(
            self.cluster_servlet, "set_cluster_config", config
        )

    ##############################################
    # Auth cache internal functions
    ##############################################
    def add_user_to_auth_cache(self, token, refresh_cache=True):
        return self.call_actor_method(
            self.cluster_servlet, "add_user_to_auth_cache", token, refresh_cache
        )

    def resource_access_level(self, token_hash: str, resource_uri: str):
        return self.call_actor_method(
            self.cluster_servlet,
            "resource_access_level",
            token_hash,
            resource_uri,
        )

    def user_resources(self, token_hash: str):
        return self.call_actor_method(
            self.cluster_servlet, "user_resources", token_hash
        )

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
        return self.call_actor_method(
            self.cluster_servlet, "clear_auth_cache", token_hash
        )

    ##############################################
    # Key to servlet where it is stored mapping
    ##############################################
    def mark_env_servlet_name_as_initialized(self, env_servlet_name: str):
        return self.call_actor_method(
            self.cluster_servlet,
            "mark_env_servlet_name_as_initialized",
            env_servlet_name,
        )

    def is_env_servlet_name_initialized(self, env_servlet_name: str) -> bool:
        return self.call_actor_method(
            self.cluster_servlet, "is_env_servlet_name_initialized", env_servlet_name
        )

    def get_all_initialized_env_servlet_names(self) -> Set[str]:
        return list(
            self.call_actor_method(
                self.cluster_servlet,
                "get_all_initialized_env_servlet_names",
            )
        )

    def get_env_servlet_name_for_key(self, key: Any):
        return self.call_actor_method(
            self.cluster_servlet, "get_env_servlet_name_for_key", key
        )

    def _put_env_servlet_name_for_key(self, key: Any, env_servlet_name: str):
        return self.call_actor_method(
            self.cluster_servlet, "put_env_servlet_name_for_key", key, env_servlet_name
        )

    def _pop_env_servlet_name_for_key(self, key: Any, *args) -> str:
        return self.call_actor_method(
            self.cluster_servlet, "pop_env_servlet_name_for_key", key, *args
        )

    ##############################################
    # KV Store: Keys
    ##############################################
    @staticmethod
    def keys_for_env_servlet_name(env_servlet_name: str) -> List[Any]:
        return ObjStore.call_actor_method(
            ObjStore.get_env_servlet(env_servlet_name), "keys_local"
        )

    def keys_local(self) -> List[Any]:
        if self.has_local_storage:
            return list(self._kv_store.keys())
        else:
            return []

    def keys(self) -> List[Any]:
        # Return keys across the cluster, not only in this process
        return self.call_actor_method(
            self.cluster_servlet, "get_key_to_env_servlet_name_dict_keys"
        )

    ##############################################
    # KV Store: Put
    ##############################################
    @staticmethod
    def put_for_env_servlet_name(env_servlet_name: str, key: Any, value: Any):
        logger.info(f"Putting {key} and {value} into servlet {env_servlet_name}")
        return ObjStore.call_actor_method(
            ObjStore.get_env_servlet(env_servlet_name), "put_local", key, value
        )

    def put_local(self, key: Any, value: Any):
        if self.has_local_storage:
            self._kv_store[key] = value
            self._put_env_servlet_name_for_key(key, self.servlet_name)
        else:
            raise NoLocalObjStoreError()

    def put(self, key: Any, value: Any, env: str = None):
        # Before replacing something else, check if this op will even be valid.
        if env is None and not self.servlet_name:
            raise NoLocalObjStoreError()

        if env is not None and self.get_env_servlet(env) is None:
            raise ObjStoreError(
                f"Env {env} does not exist; cannot put key {key} there."
            )

        # If it does exist somewhere, no more!
        if self.get(key, default=None) is not None:
            logger.warning("Key already exists in some env, overwriting.")
            self.pop(key)

        # If env is None, write to our own servlet, either via local or via global KV store
        env = env or self.servlet_name
        if self.has_local_storage and env == self.servlet_name:
            self.put_local(key, value)
        else:
            self.put_for_env_servlet_name(env, key, value)

    ##############################################
    # KV Store: Get
    ##############################################
    @staticmethod
    def get_from_env_servlet_name(
        env_servlet_name: str, key: Any, default: Optional[Any] = None
    ):
        logger.info(f"Getting {key} from servlet {env_servlet_name}")
        return ObjStore.call_actor_method(
            ObjStore.get_env_servlet(env_servlet_name), "get_local", key, default
        )

    def get_local(self, key: Any, default: Optional[Any] = None):
        if self.has_local_storage:
            try:
                return self._kv_store[key]
            except KeyError as e:
                if default == KeyError:
                    raise e
                return default
        else:
            if default == KeyError:
                raise KeyError(f"No local store exists; key {key} not found.")
            return default

    def get(
        self,
        key: Any,
        default: Optional[Any] = None,
        check_other_envs: bool = True,
    ):
        # First check if it's in the Python kv store
        try:
            return self.get_local(key, default=KeyError)
        except KeyError as e:
            key_err = e

        if not check_other_envs:
            if default == KeyError:
                raise key_err
            return default

        # If not, check if it's in another env's servlet
        env_servlet_name = self.get_env_servlet_name_for_key(key)
        if env_servlet_name == self.servlet_name and self.has_local_storage:
            raise ValueError(
                "Key not found in kv store despite env servlet specifying that it is here."
            )

        if env_servlet_name is None:
            if default == KeyError:
                raise key_err
            return default

        try:
            return self.get_from_env_servlet_name(
                env_servlet_name, key, default=KeyError
            )
        except KeyError:
            raise ObjStoreError(
                f"Key was supposed to be in {env_servlet_name}, but it was not found there."
            )

    ##############################################
    # KV Store: Contains
    ##############################################
    @staticmethod
    def contains_for_env_servlet_name(env_servlet_name: str, key: Any):
        return ObjStore.call_actor_method(
            ObjStore.get_env_servlet(env_servlet_name), "contains_local", key
        )

    def contains_local(self, key: Any):
        if self.has_local_storage:
            return key in self._kv_store
        else:
            return False

    def contains(self, key: Any):
        if self.contains_local(key):
            return True

        env_servlet_name = self.get_env_servlet_name_for_key(key)
        if env_servlet_name == self.servlet_name and self.has_local_storage:
            raise ObjStoreError(
                "Key not found in kv store despite env servlet specifying that it is here."
            )

        if env_servlet_name is None:
            return False

        return self.contains_for_env_servlet_name(env_servlet_name, key)

    ##############################################
    # KV Store: Pop
    ##############################################
    @staticmethod
    def pop_from_env_servlet_name(env_servlet_name: str, key: Any, *args) -> Any:
        return ObjStore.call_actor_method(
            ObjStore.get_env_servlet(env_servlet_name), "pop_local", key, *args
        )

    def pop_local(self, key: Any, *args) -> Any:
        if self.has_local_storage:
            try:
                res = self._kv_store.pop(key)
            except KeyError as key_err:
                # Return the default if it was provided, else raise the error as expected
                if args:
                    return args[0]
                else:
                    raise key_err

            # If the key was found in this env, we also need to pop it
            # from the global env for key cache.
            env_name = self._pop_env_servlet_name_for_key(key, None)
            if env_name and env_name != self.servlet_name:
                raise ObjStoreError(
                    "The key was popped from this env, but the global env for key cache says it's in another one."
                )

            return res
        else:
            if args:
                return args[0]
            else:
                raise KeyError(f"No local store exists; key {key} not found.")

    def pop(self, key: Any, *args) -> Any:
        try:
            return self.pop_local(key)
        except KeyError as e:
            key_err = e

        # The key was not found in this env
        # So, we check the global key to env cache to see if it's elsewhere
        env_servlet_name = self.get_env_servlet_name_for_key(key)
        if env_servlet_name:
            if env_servlet_name == self.servlet_name and self.has_local_storage:
                raise ObjStoreError(
                    "The key was not found in this env, but the global env for key cache says it's here."
                )
            else:
                # The key was found in another env, so we need to pop it from there
                return self.pop_from_env_servlet_name(env_servlet_name, key)
        else:
            # Was not found in any env
            if args:
                return args[0]
            else:
                raise key_err

    ##############################################
    # KV Store: Delete
    ##############################################
    @staticmethod
    def delete_for_env_servlet_name(env_servlet_name: str, key: Any):
        return ObjStore.call_actor_method(
            ObjStore.get_env_servlet(env_servlet_name), "delete_local", key
        )

    def delete_local(self, key: Any):
        self.pop_local(key)

    def delete(self, key: Union[Any, List[Any]]):
        keys_to_delete = [key] if isinstance(key, str) else key
        for key_to_delete in keys_to_delete:
            if self.contains_local(key_to_delete):
                self.delete_local(key_to_delete)
            else:
                env_servlet_name = self.get_env_servlet_name_for_key(key_to_delete)
                if env_servlet_name == self.servlet_name and self.has_local_storage:
                    raise ObjStoreError(
                        "Key not found in kv store despite env servlet specifying that it is here."
                    )
                if env_servlet_name is None:
                    raise KeyError(f"Key {key} not found in any env.")

                self.delete_for_env_servlet_name(env_servlet_name, key_to_delete)

    ##############################################
    # KV Store: Clear
    ##############################################
    @staticmethod
    def clear_for_env_servlet_name(env_servlet_name: str):
        return ObjStore.call_actor_method(
            ObjStore.get_env_servlet(env_servlet_name), "clear_local"
        )

    def clear_local(self):
        if self.has_local_storage:
            for k in list(self._kv_store.keys()):
                # Pop handles removing from global obj store vs local one
                self.pop_local(k)

    def clear(self):
        logger.warning("Clearing all keys from all envs in the object store!")
        for env_servlet_name in self.get_all_initialized_env_servlet_names():
            if env_servlet_name == self.servlet_name and self.has_local_storage:
                self.clear_local()
            else:
                self.clear_for_env_servlet_name(env_servlet_name)

    ##############################################
    # KV Store: Rename
    ##############################################
    def rename(self, old_key: Any, new_key: Any):
        # We also need to rename the resource itself
        env_servlet_name_containing_old_key = self.get_env_servlet_name_for_key(old_key)
        obj = self.pop(old_key)
        if obj is not None and hasattr(obj, "rns_address"):
            # Note - we set the obj.name here so the new_key is correctly turned into an rns_address, whether its
            # a full address or just a name. Then, the new_key is set to just the name so its store properly in the
            # kv store.
            obj.name = new_key  # new_key can be an rns_address! e.g. if called by Module.rename
            new_key = obj.name  # new_key is now just the name

        # By passing default, we don't throw an error if the key is not found
        self.put(new_key, obj, env=env_servlet_name_containing_old_key)

    ##############################################
    # Get several keys for function initialization utiliies
    ##############################################
    def get_list(self, keys: List[str], default: Optional[Any] = None):
        return [self.get(key, default=default or key) for key in keys]

    def get_obj_refs_list(self, keys: List[Any]):
        return [
            self.get(key, default=key) if isinstance(key, str) else key for key in keys
        ]

    def get_obj_refs_dict(self, d: Dict[Any, Any]):
        return {
            k: self.get(v, default=v) if isinstance(v, str) else v for k, v in d.items()
        }
