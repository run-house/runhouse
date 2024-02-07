import logging
import os
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Union

import ray

import runhouse

logger = logging.getLogger(__name__)


class RaySetupOption(str, Enum):
    GET_OR_FAIL = "get_or_fail"
    TEST_PROCESS = "test_process"


class ClusterServletSetupOption(str, Enum):
    GET_OR_CREATE = "get_or_create"
    GET_OR_FAIL = "get_or_fail"
    FORCE_CREATE = "force_create"


class ObjStoreError(Exception):
    pass


class NoLocalObjStoreError(ObjStoreError):
    def __init__(self):
        super().__init__("No local object store exists; cannot perform operation.")


def get_cluster_servlet(create_if_not_exists: bool = False):
    from runhouse.servers.cluster_servlet import ClusterServlet

    if not ray.is_initialized():
        raise ConnectionError("Ray is not initialized.")

    # Previously used list_actors here to avoid a try/except, but it is finicky
    # when there are several Ray clusters running. In tests, we typically run multiple
    # clusters, so let's avoid this.
    try:
        cluster_servlet = ray.get_actor("cluster_servlet", namespace="runhouse")
    except ValueError:
        cluster_servlet = None

    if cluster_servlet is None and create_if_not_exists:
        cluster_servlet = (
            ray.remote(ClusterServlet)
            .options(
                name="cluster_servlet",
                get_if_exists=True,
                lifetime="detached",
                namespace="runhouse",
            )
            .remote()
        )

        # Make sure cluster servlet is actually initialized
        ray.get(cluster_servlet.get_cluster_config.remote())

    return cluster_servlet


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
        self.installed_envs = {}  # TODO: consider deleting it?
        self._kv_store: Dict[Any, Any] = None

    def initialize(
        self,
        servlet_name: Optional[str] = None,
        has_local_storage: bool = False,
        setup_ray: RaySetupOption = RaySetupOption.GET_OR_FAIL,
        ray_address: str = "auto",
        setup_cluster_servlet: ClusterServletSetupOption = ClusterServletSetupOption.GET_OR_CREATE,
    ):
        # The initialization of the obj_store needs to be in a separate method
        # so the HTTPServer actually initalizes the obj_store,
        # and it doesn't get created and destroyed when
        # nginx runs http_server.py as a module.

        # ClusterServlet essentially functions as a global state/metadata store
        # for all nodes connected to this Ray cluster.
        from runhouse.resources.hardware.ray_utils import kill_actors

        # Only if ray is not initialized do we attempt a setup process.
        if not ray.is_initialized():
            if setup_ray == RaySetupOption.TEST_PROCESS:
                # When we run ray.init() with no address provided
                # and no Ray is running, it will start a new Ray cluster,
                # but one that is only exposed to this process. This allows us to
                # run unit tests without starting bare metal Ray clusters on each machine.
                ray.init(
                    ignore_reinit_error=True,
                    logging_level=logging.ERROR,
                    namespace="runhouse",
                )
            else:
                ray.init(
                    address=ray_address,
                    ignore_reinit_error=True,
                    logging_level=logging.ERROR,
                    namespace="runhouse",
                )

        # Now, we expect to be connected to an initialized Ray instance.
        if setup_cluster_servlet == ClusterServletSetupOption.FORCE_CREATE:
            kill_actors(namespace="runhouse", gracefully=False)

        create_if_not_exists = (
            setup_cluster_servlet != ClusterServletSetupOption.GET_OR_FAIL
        )
        self.cluster_servlet = get_cluster_servlet(
            create_if_not_exists=create_if_not_exists
        )
        if self.cluster_servlet is None:
            # TODO: logger.<method> is not printing correctly here when doing `runhouse start`.
            # Fix this and general logging.
            logging.warning(
                "Warning, cluster servlet is not initialized. Object Store operations will not work."
            )

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
    def get_env_servlet(
        env_name: str,
        create: bool = False,
        raise_ex_if_not_found: bool = False,
        resources: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        # Need to import these here to avoid circular imports
        from runhouse.globals import env_servlets
        from runhouse.servers.env_servlet import EnvServlet

        if env_name in env_servlets:
            return env_servlets[env_name]

        # It may not have been cached, but does exist
        try:
            existing_actor = ray.get_actor(env_name, namespace="runhouse")
            env_servlets[env_name] = existing_actor
            return existing_actor
        except ValueError:
            # ValueError: Failed to look up actor with name ...
            pass

        if resources:
            # Check if requested resources are available
            available_resources = ray.available_resources()
            for k, v in resources.items():
                if k not in available_resources or available_resources[k] < v:
                    raise Exception(
                        f"Requested resource {k}={v} is not available on the cluster. "
                        f"Available resources: {available_resources}"
                    )
        else:
            resources = {}

        # Otherwise, create it
        if create:
            new_env_actor = (
                ray.remote(EnvServlet)
                .options(
                    name=env_name,
                    get_if_exists=True,
                    runtime_env=kwargs["runtime_env"]
                    if "runtime_env" in kwargs
                    else None,
                    num_cpus=resources.pop("CPU", None),
                    num_gpus=resources.pop("GPU", None),
                    resources=resources,
                    lifetime="detached",
                    namespace="runhouse",
                    max_concurrency=1000,
                )
                .remote(env_name=env_name)
            )

            # Make sure env_servlet is actually initialized
            # ray.get(new_env_actor.register_activity.remote())

            env_servlets[env_name] = new_env_actor
            return new_env_actor

        else:
            if raise_ex_if_not_found:
                raise ObjStoreError(
                    f"Environment {env_name} does not exist. Please send it to the cluster first."
                )
            else:
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

    def set_cluster_config_value(self, key: str, value: Any):
        return self.call_actor_method(
            self.cluster_servlet, "set_cluster_config_value", key, value
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
        from runhouse.globals import configs
        from runhouse.rns.utils.api import ResourceAccess
        from runhouse.servers.http.auth import hash_token
        from runhouse.servers.http.http_utils import load_current_cluster_rns_address

        if token_hash is None:
            # If no token is provided assume no access
            return False

        # The logged-in user always has full access to the cluster and its resources. This is especially
        # important if they flip on Den Auth without saving the cluster.
        if configs.token and hash_token(configs.token) == token_hash:
            return True

        cluster_uri = load_current_cluster_rns_address()
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
    # Remove Env Servlet
    ##############################################
    def remove_env_servlet_name(self, env_servlet_name: str):
        return self.call_actor_method(
            self.cluster_servlet, "remove_env_servlet_name", env_servlet_name
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
    def put_for_env_servlet_name(
        env_servlet_name: str, key: Any, data: Any, serialization: Optional[str] = None
    ):
        return ObjStore.call_actor_method(
            ObjStore.get_env_servlet(env_servlet_name),
            "put_local",
            key,
            data=data,
            serialization=serialization,
        )

    def put_local(self, key: Any, value: Any):
        if self.has_local_storage:
            self._kv_store[key] = value
            self._put_env_servlet_name_for_key(key, self.servlet_name)
        else:
            raise NoLocalObjStoreError()

    def put(
        self,
        key: Any,
        value: Any,
        env: Optional[str] = None,
        serialization: Optional[str] = None,
        create_env_if_not_exists: bool = False,
    ):
        # Before replacing something else, check if this op will even be valid.
        if env is None and self.servlet_name is None:
            raise NoLocalObjStoreError()

        # If it was not specified, we want to put into our own servlet_name
        env = env or self.servlet_name

        if self.get_env_servlet(env) is None:
            if create_env_if_not_exists:
                self.get_env_servlet(env, create=True)
            else:
                raise ObjStoreError(
                    f"Env {env} does not exist; cannot put key {key} there."
                )

        # If it does exist somewhere, no more!
        if self.get(key, default=None) is not None:
            logger.warning("Key already exists in some env, overwriting.")
            self.pop(key)

        if self.has_local_storage and env == self.servlet_name:
            if serialization is not None:
                raise ObjStoreError(
                    "We should never reach this branch if serialization is not None."
                )
            self.put_local(key, value)
        else:
            self.put_for_env_servlet_name(env, key, value, serialization)

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
    def pop_from_env_servlet_name(
        env_servlet_name: str, key: Any, serialization: Optional[str] = "pickle", *args
    ) -> Any:
        return ObjStore.call_actor_method(
            ObjStore.get_env_servlet(env_servlet_name),
            "pop_local",
            key,
            serialization,
            *args,
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

    def pop(self, key: Any, serialization: Optional[str] = "pickle", *args) -> Any:
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
                return self.pop_from_env_servlet_name(
                    env_servlet_name, key, serialization
                )
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

    def _delete_env_contents(self, env_name: Any):
        from runhouse.globals import env_servlets

        # clear keys in the env servlet
        deleted_keys = self.keys_for_env_servlet_name(env_name)
        self.clear_for_env_servlet_name(env_name)

        # delete the env servlet actor and remove its references
        if env_name in env_servlets:
            actor = env_servlets[env_name]
            ray.kill(actor)

            del env_servlets[env_name]
        self.remove_env_servlet_name(env_name)

        return deleted_keys

    def delete(self, key: Union[Any, List[Any]]):
        keys_to_delete = [key] if isinstance(key, str) else key
        deleted_keys = []

        for key_to_delete in keys_to_delete:
            if key_to_delete in self.get_all_initialized_env_servlet_names():
                deleted_keys += self._delete_env_contents(key_to_delete)

            if key_to_delete in deleted_keys:
                continue

            if self.contains_local(key_to_delete):
                self.delete_local(key_to_delete)
                deleted_keys.append(key_to_delete)
            else:
                env_servlet_name = self.get_env_servlet_name_for_key(key_to_delete)
                if env_servlet_name == self.servlet_name and self.has_local_storage:
                    raise ObjStoreError(
                        "Key not found in kv store despite env servlet specifying that it is here."
                    )
                if env_servlet_name is None:
                    raise KeyError(f"Key {key} not found in any env.")

                self.delete_for_env_servlet_name(env_servlet_name, key_to_delete)
                deleted_keys.append(key_to_delete)

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
    @staticmethod
    def rename_for_env_servlet_name(env_servlet_name: str, old_key: Any, new_key: Any):
        return ObjStore.call_actor_method(
            ObjStore.get_env_servlet(env_servlet_name),
            "rename_local",
            old_key,
            new_key,
        )

    def rename_local(self, old_key: Any, new_key: Any):
        if self.servlet_name is None or not self.has_local_storage:
            raise NoLocalObjStoreError()

        obj = self.pop(old_key)
        if obj is not None and hasattr(obj, "rns_address"):
            # Note - we set the obj.name here so the new_key is correctly turned into an rns_address, whether its
            # a full address or just a name. Then, the new_key is set to just the name so its store properly in the
            # kv store.
            obj.name = new_key  # new_key can be an rns_address! e.g. if called by Module.rename
            new_key = obj.name  # new_key is now just the name

        # By passing default, we don't throw an error if the key is not found
        self.put(new_key, obj, env=self.servlet_name)

    def rename(self, old_key: Any, new_key: Any):
        # We also need to rename the resource itself
        env_servlet_name_containing_old_key = self.get_env_servlet_name_for_key(old_key)
        if (
            env_servlet_name_containing_old_key == self.servlet_name
            and self.has_local_storage
        ):
            self.rename_local(old_key, new_key)
        else:
            self.rename_for_env_servlet_name(
                env_servlet_name_containing_old_key, old_key, new_key
            )

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

    ##############################################
    # More specific helpers
    ##############################################
    def put_resource(
        self,
        serialized_data: Any,
        serialization: Optional[str] = None,
        env_name: Optional[str] = None,
    ) -> "Response":
        from runhouse.servers.http.http_utils import deserialize_data

        if env_name is None and self.servlet_name is None:
            raise ObjStoreError("No env name provided and no servlet name set.")

        env_name = env_name or self.servlet_name
        if self.has_local_storage and env_name == self.servlet_name:
            resource_config, state, dryrun = deserialize_data(
                serialized_data, serialization
            )
            return self.put_resource_local(resource_config, state, dryrun)

        # Normally, serialization and deserialization happens within the servlet
        # However, if we're putting an env, we need to deserialize it here and
        # actually create the corresponding env servlet.
        resource_config, _, _ = deserialize_data(serialized_data, serialization)
        if resource_config["resource_type"] == "env":

            # Note that the passed in `env_name` and the `env_name_to_create` here are
            # distinct. The `env_name` is the name of the env servlet where we want to store
            # the resource itself. The `env_name_to_create` is the name of the env servlet
            # that we need to create because we are putting an env resource somewhere on the cluster.
            env_name_to_create = resource_config["env_name"]
            runtime_env = (
                {"conda_env": env_name_to_create}
                if resource_config["resource_subtype"] == "CondaEnv"
                else {}
            )

            _ = ObjStore.get_env_servlet(
                env_name=env_name_to_create,
                create=True,
                runtime_env=runtime_env,
                resources=resource_config.get("compute", None),
            )

        return ObjStore.call_actor_method(
            ObjStore.get_env_servlet(env_name),
            "put_resource_local",
            data=serialized_data,
            serialization=serialization,
        )

    def put_resource_local(
        self,
        resource_config: Dict[str, Any],
        state: Dict[Any, Any],
        dryrun: bool,
    ) -> str:
        from runhouse.resources.module import Module
        from runhouse.resources.resource import Resource
        from runhouse.rns.utils.names import _generate_default_name

        state = state or {}
        # Resolve any sub-resources which are string references to resources already sent to this cluster.
        # We need to pop the resource's own name so it doesn't get resolved if it's already present in the
        # obj_store.
        name = resource_config.pop("name")
        subtype = resource_config.pop("resource_subtype")
        provider = (
            resource_config.pop("provider") if "provider" in resource_config else None
        )

        resource_config = self.get_obj_refs_dict(resource_config)
        resource_config["name"] = name
        resource_config["resource_subtype"] = subtype
        if provider:
            resource_config["provider"] = provider

        logger.info(
            f"Message received from client to construct resource: {resource_config}"
        )

        resource = Resource.from_config(config=resource_config, dryrun=dryrun)

        for attr, val in state.items():
            setattr(resource, attr, val)

        name = resource.name or _generate_default_name(prefix=resource.RESOURCE_TYPE)
        if isinstance(resource, Module):
            resource.rename(name)
        else:
            resource.name = name

        self.put(resource.name, resource)

        # Return the name in case we had to set it
        return resource.name

    ##############################################
    # Cluster info methods
    ##############################################

    def get_status(self):
        config_cluster = self.get_cluster_config()
        envs_in_cluster = self.get_all_initialized_env_servlet_names()
        cluster_servlets = {}
        for env in envs_in_cluster:
            env_keys = self.keys_for_env_servlet_name(env)
            resources_in_env = self.get_list(env_keys)
            resources_in_env_modified = []

            # The objects in env can be of any type, and not only runhouse resources,
            # therefore we need to distinguish them when creating the list of the resources in each env.
            for r in resources_in_env:
                cls = type(r)
                py_module = cls.__module__
                cls_name = (
                    cls.__qualname__
                    if py_module == "builtins"
                    else (py_module + "." + cls.__qualname__)
                )
                if isinstance(r, runhouse.Resource):
                    resources_in_env_modified.append(
                        {"name": r.name, "resource_type": cls_name}
                    )
                else:
                    resources_in_env_modified.append(
                        {"name": r, "resource_type": cls_name}
                    )

            cluster_servlets[env] = resources_in_env_modified
        config_cluster["envs"] = cluster_servlets
        return config_cluster
