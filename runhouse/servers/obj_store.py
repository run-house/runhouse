import asyncio
import copy
import inspect
import logging
import os
import time
import uuid
from enum import Enum
from functools import wraps
from typing import Any, Dict, List, Optional, Set, Union

import ray
from pydantic import BaseModel

from runhouse.logger import get_logger

from runhouse.rns.defaults import req_ctx
from runhouse.rns.utils.api import ResourceVisibility
from runhouse.utils import (
    arun_in_thread,
    generate_default_name,
    LogToFolder,
    sync_function,
)

logger = get_logger(__name__)


class RaySetupOption(str, Enum):
    GET_OR_FAIL = "get_or_fail"
    TEST_PROCESS = "test_process"


class ClusterServletSetupOption(str, Enum):
    GET_OR_CREATE = "get_or_create"
    GET_OR_FAIL = "get_or_fail"
    FORCE_CREATE = "force_create"


class ObjStoreError(Exception):
    pass


class RunhouseStopIteration(Exception):
    pass


class ActiveFunctionCallInfo(BaseModel):
    key: str
    method_name: str
    request_id: str
    start_time: float


class NoLocalObjStoreError(ObjStoreError):
    def __init__(self, *args):
        super().__init__("No local object store exists; cannot perform operation.")


def get_cluster_servlet(
    create_if_not_exists: bool = False,
    runtime_env: Optional[Dict] = None,
):
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

    # Ensure the ClusterServlet starts on the head node, per
    # https://discuss.ray.io/t/how-to-ensure-actor-is-running-on-the-same-node-only/2083/3
    current_ip = ray.get_runtime_context().worker.node_ip_address
    if cluster_servlet is None and create_if_not_exists:
        cluster_servlet = (
            ray.remote(ClusterServlet)
            .options(
                name="cluster_servlet",
                get_if_exists=True,
                lifetime="detached",
                namespace="runhouse",
                max_concurrency=1000,
                resources={f"node:{current_ip}": 0.001},
                num_cpus=0,
                runtime_env=runtime_env,
            )
            .remote()
        )

        # Make sure cluster servlet is actually initialized
        ray.get(cluster_servlet.aget_cluster_config.remote())

    return cluster_servlet


def context_wrapper(func):
    @wraps(func)
    async def wrapper(self, *args, **kwargs):
        ctx_token = None
        try:
            if not req_ctx.get():
                ctx_token = await self.apopulate_ctx_locally()

            res = await func(self, *args, **kwargs)
        except Exception as e:
            raise e
        finally:
            if ctx_token:
                self.unset_ctx(ctx_token)

        return res

    return wrapper


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
        self.cluster_config: Optional[Dict[str, Any]] = None
        self.imported_modules = {}
        self.installed_envs = {}  # TODO: consider deleting it?
        self._kv_store: Dict[Any, Any] = None
        self.env_servlet_cache = {}
        self.active_function_calls = {}

    async def ainitialize(
        self,
        servlet_name: Optional[str] = None,
        has_local_storage: bool = False,
        setup_ray: RaySetupOption = RaySetupOption.GET_OR_FAIL,
        ray_address: str = "auto",
        setup_cluster_servlet: ClusterServletSetupOption = ClusterServletSetupOption.GET_OR_CREATE,
        runtime_env: Optional[Dict] = None,
    ):
        # The initialization of the obj_store needs to be in a separate method
        # so the HTTPServer actually initalizes the obj_store,
        # and it doesn't get created and destroyed when
        # caddy runs http_server.py as a module.

        # ClusterServlet essentially functions as a global state/metadata store
        # for all nodes connected to this Ray cluster.

        # If the servlet name is already set, the obj_store has already been initialized
        if self.servlet_name is not None:
            return

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
            create_if_not_exists=create_if_not_exists,
            runtime_env=runtime_env,
        )
        if self.cluster_servlet is None:
            # TODO: logger.<method> is not printing correctly here when doing `runhouse start`.
            # Fix this and general logging.
            logger.warning(
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
            if await self.ais_env_servlet_name_initialized(servlet_name):
                raise ValueError(
                    f"There already exists an EnvServlet with name {servlet_name}."
                )
            else:
                await self.amark_env_servlet_name_as_initialized(servlet_name)

        self.servlet_name = servlet_name
        self.has_local_storage = has_local_storage
        if self.has_local_storage:
            self._kv_store = {}

        # Store a local copy of the cluster_config here
        self.cluster_config = await self.aget_cluster_config(refresh=True)

        num_gpus = ray.cluster_resources().get("GPU", 0)
        cuda_visible_devices = list(range(int(num_gpus)))
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, cuda_visible_devices))

    def initialize(
        self,
        servlet_name: Optional[str] = None,
        has_local_storage: bool = False,
        setup_ray: RaySetupOption = RaySetupOption.GET_OR_FAIL,
        ray_address: str = "auto",
        setup_cluster_servlet: ClusterServletSetupOption = ClusterServletSetupOption.GET_OR_CREATE,
        runtime_env: Optional[Dict] = None,
    ):
        return sync_function(self.ainitialize)(
            servlet_name,
            has_local_storage,
            setup_ray,
            ray_address,
            setup_cluster_servlet,
            runtime_env,
        )

    def get_process_env(self) -> Optional["Env"]:
        """
        If this is an env servlet object store, then we are within a Runhouse env.
        Return the env so it can be used for Runhouse primitives.
        """
        if self.servlet_name is not None and self.has_local_storage:
            # Each env is stored within itself, I believe
            return self.get(self.servlet_name)

    def get_internal_ips(self):
        """Get list of internal IPs of all nodes in the cluster."""
        cluster_config = self.get_cluster_config()
        if "stable_internal_external_ips" in cluster_config:
            return [
                internal_ip
                for internal_ip, external_ip in cluster_config[
                    "stable_internal_external_ips"
                ]
            ]
        else:
            if not ray.is_initialized():
                raise ConnectionError("Ray is not initialized.")

            cluster_nodes = ray.nodes()
            return [node["NodeManagerAddress"] for node in cluster_nodes]

    ##############################################
    # Generic helpers
    ##############################################
    async def acall_env_servlet_method(
        self,
        servlet_name: str,
        method: str,
        *args,
        use_env_servlet_cache: bool = True,
        **kwargs,
    ):
        env_servlet = self.get_env_servlet(
            servlet_name, use_env_servlet_cache=use_env_servlet_cache
        )
        if env_servlet is None:
            raise ObjStoreError(
                f"Got None env_servlet for servlet_name {servlet_name}."
            )
        try:
            return await ObjStore.acall_actor_method(
                env_servlet, method, *args, **kwargs
            )
        except (ray.exceptions.RayActorError, ray.exceptions.OutOfMemoryError) as e:
            if isinstance(
                e, ray.exceptions.OutOfMemoryError
            ) or "died unexpectedly before finishing this task" in str(e):
                await self.adelete_env_contents(servlet_name)

            raise e

    @staticmethod
    async def acall_actor_method(
        actor: ray.actor.ActorHandle, method: str, *args, **kwargs
    ):
        if actor is None:
            raise ObjStoreError("Attempting to call an actor method on a None actor.")
        return await getattr(actor, method).remote(*args, **kwargs)

    @staticmethod
    def call_actor_method(actor: ray.actor.ActorHandle, method: str, *args, **kwargs):
        if actor is None:
            raise ObjStoreError("Attempting to call an actor method on a None actor.")

        return ray.get(getattr(actor, method).remote(*args, **kwargs))

    def get_env_servlet(
        self,
        env_name: str,
        create: bool = False,
        raise_ex_if_not_found: bool = False,
        resources: Optional[Dict[str, Any]] = None,
        use_env_servlet_cache: bool = True,
        **kwargs,
    ):
        # Need to import these here to avoid circular imports
        from runhouse.servers.env_servlet import EnvServlet

        if use_env_servlet_cache and env_name in self.env_servlet_cache:
            return self.env_servlet_cache[env_name]

        # It may not have been cached, but does exist
        try:
            existing_actor = ray.get_actor(env_name, namespace="runhouse")
            if use_env_servlet_cache:
                self.env_servlet_cache[env_name] = existing_actor
            return existing_actor
        except ValueError:
            # ValueError: Failed to look up actor with name ...
            pass

        # Otherwise, create it
        if create:
            if resources is None:
                resources = {}

            if "node_idx" in resources and ("CPU" in resources or "GPU" in resources):
                raise ValueError(
                    "Cannot specify both node_idx and CPU/GPU resources for an env."
                )

            # Replace node_idx with actual node IP in Ray resources request
            node_idx = resources.pop("node_idx", None)
            if node_idx is not None:
                cluster_ips = self.get_internal_ips()
                if node_idx >= len(cluster_ips):
                    raise ValueError(
                        f"Node index {node_idx} is out of bounds for cluster with {len(cluster_ips)} nodes."
                    )
                resources[f"node:{cluster_ips[node_idx]}"] = 0.001

            # Check if requested resources are available
            available_resources = ray.available_resources()
            for k, v in resources.items():
                if k not in available_resources or available_resources[k] < v:
                    raise Exception(
                        f"Requested resource {k}={v} is not available on the cluster. "
                        f"Available resources: {available_resources}"
                    )

            new_env_actor = (
                ray.remote(EnvServlet)
                .options(
                    name=env_name,
                    get_if_exists=True,
                    runtime_env=kwargs["runtime_env"]
                    if "runtime_env" in kwargs
                    else None,
                    # Default to 0 CPUs if not specified, Ray will default it to 1
                    num_cpus=resources.pop("CPU", 0),
                    num_gpus=resources.pop("GPU", None),
                    memory=resources.pop("memory", None),
                    resources=resources,
                    lifetime="detached",
                    namespace="runhouse",
                    max_concurrency=1000,
                )
                .remote(env_name=env_name)
            )

            # Make sure env_servlet is actually initialized
            # ray.get(new_env_actor.register_activity.remote())
            if use_env_servlet_cache:
                self.env_servlet_cache[env_name] = new_env_actor
            return new_env_actor

        else:
            if raise_ex_if_not_found:
                raise ObjStoreError(
                    f"Environment {env_name} does not exist. Please send it to the cluster first."
                )
            else:
                return None

    @staticmethod
    def set_ctx(**ctx_args):
        from runhouse.servers.http.http_utils import RequestContext

        ctx = RequestContext(**ctx_args)
        return req_ctx.set(ctx)

    async def apopulate_ctx_locally(self):
        from runhouse.globals import configs

        token = configs.token
        return self.set_ctx(request_id=str(uuid.uuid4()), token=token)

    @staticmethod
    def unset_ctx(ctx_token):
        req_ctx.reset(ctx_token)

    ##############################################
    # Propagate den auth
    ##############################################
    # TODO: Maybe this function needs to be synchronous to propagate Den Auth changes immediately?
    # Guess we'll find out
    def is_den_auth_enabled(self):
        return (
            self.cluster_config.get("den_auth", False)
            if self.cluster_config is not None
            else False
        )

    async def aset_den_auth(self, den_auth: bool):
        await self.aset_cluster_config_value("den_auth", den_auth)

    async def aenable_den_auth(self):
        await self.aset_den_auth(True)

    async def adisable_den_auth(self):
        await self.aset_den_auth(False)

    ##############################################
    # Cluster config state storage methods
    ##############################################
    async def aget_cluster_config(self, refresh: bool = False):
        if refresh or not self.cluster_config:
            if self.cluster_servlet is not None:
                self.cluster_config = await self.acall_actor_method(
                    self.cluster_servlet, "aget_cluster_config"
                )
            else:
                return {}

        return self.cluster_config

    def get_cluster_config(self):
        return sync_function(self.aget_cluster_config)()

    async def aset_cluster_config(self, config: Dict[str, Any]):
        self.cluster_config = await self.acall_actor_method(
            self.cluster_servlet, "aset_cluster_config", config
        )

    async def aset_cluster_config_value(self, key: str, value: Any):
        self.cluster_config = await self.acall_actor_method(
            self.cluster_servlet, "aset_cluster_config_value", key, value
        )

    def set_cluster_config_value(self, key: str, value: Any):
        sync_function(self.aset_cluster_config_value)(key, value)

    ##############################################
    # Auth cache internal functions
    ##############################################
    async def aresource_access_level(self, token: str, resource_uri: str):
        return await self.acall_actor_method(
            self.cluster_servlet,
            "aresource_access_level",
            token,
            resource_uri,
        )

    def resource_access_level(self, token: str, resource_uri: str):
        return sync_function(self.aresource_access_level)(token, resource_uri)

    async def aget_username(self, token: str):
        return await self.acall_actor_method(
            self.cluster_servlet, "aget_username", token
        )

    async def ahas_resource_access(self, token: str, resource_uri=None) -> bool:
        """Checks whether user has read or write access to a given module saved on the cluster."""
        from runhouse.globals import configs, rns_client
        from runhouse.rns.utils.api import ResourceAccess
        from runhouse.servers.http.http_utils import load_current_cluster_rns_address

        if token is None:
            # If no token is provided assume no access
            return False

        # The logged-in user always has full access to the cluster and its resources. This is especially
        # important if they flip on Den Auth without saving the cluster.

        # configs.token is the token stored on the cluster itself, which is itself a hashed subtoken
        config_token = configs.token
        if config_token:
            if config_token == token:
                return True

            if resource_uri and rns_client.validate_cluster_token(
                cluster_token=token, cluster_uri=resource_uri
            ):
                return True

        cluster_uri = load_current_cluster_rns_address()
        cluster_access = await self.aresource_access_level(token, cluster_uri)
        if cluster_access == ResourceAccess.WRITE:
            # if user has write access to cluster will have access to all resources
            return True

        if resource_uri != cluster_uri and cluster_access == ResourceAccess.READ:
            # If the user has READ access to the cluster and this isn't a cluster management
            # endpoint, they have access to all resources
            return True

        if resource_uri is None and cluster_access not in [
            ResourceAccess.WRITE,
            ResourceAccess.READ,
        ]:
            # If module does not have a name, must have access to the cluster
            return False

        resource_access_level = await self.aresource_access_level(token, resource_uri)
        return resource_access_level in [ResourceAccess.WRITE, ResourceAccess.READ]

    async def aclear_auth_cache(self, token: str = None):
        return await self.acall_actor_method(
            self.cluster_servlet, "aclear_auth_cache", token
        )

    ##############################################
    # Key to servlet where it is stored mapping
    ##############################################
    async def amark_env_servlet_name_as_initialized(self, env_servlet_name: str):
        return await self.acall_actor_method(
            self.cluster_servlet,
            "amark_env_servlet_name_as_initialized",
            env_servlet_name,
        )

    async def ais_env_servlet_name_initialized(self, env_servlet_name: str) -> bool:
        return await self.acall_actor_method(
            self.cluster_servlet, "ais_env_servlet_name_initialized", env_servlet_name
        )

    async def aget_all_initialized_env_servlet_names(self) -> Set[str]:
        return list(
            await self.acall_actor_method(
                self.cluster_servlet,
                "aget_all_initialized_env_servlet_names",
            )
        )

    def get_all_initialized_env_servlet_names(self) -> Set[str]:
        return sync_function(self.aget_all_initialized_env_servlet_names)()

    async def aget_env_servlet_name_for_key(self, key: Any):
        return await self.acall_actor_method(
            self.cluster_servlet, "aget_env_servlet_name_for_key", key
        )

    def get_env_servlet_name_for_key(self, key: Any):
        return sync_function(self.aget_env_servlet_name_for_key)(key)

    async def _aput_env_servlet_name_for_key(self, key: Any, env_servlet_name: str):
        return await self.acall_actor_method(
            self.cluster_servlet, "aput_env_servlet_name_for_key", key, env_servlet_name
        )

    async def _apop_env_servlet_name_for_key(self, key: Any, *args) -> str:
        return await self.acall_actor_method(
            self.cluster_servlet, "apop_env_servlet_name_for_key", key, *args
        )

    ##############################################
    # Remove Env Servlet
    ##############################################
    async def aclear_all_references_to_env_servlet_name(self, env_servlet_name: str):
        return await self.acall_actor_method(
            self.cluster_servlet,
            "aclear_all_references_to_env_servlet_name",
            env_servlet_name,
        )

    ##############################################
    # KV Store: Keys
    ##############################################
    async def akeys_for_env_servlet_name(self, env_servlet_name: str) -> List[Any]:
        return await self.acall_env_servlet_method(env_servlet_name, "akeys_local")

    def keys_for_env_servlet_name(self, env_servlet_name: str) -> List[Any]:
        return sync_function(self.akeys_for_env_servlet_name)(env_servlet_name)

    def keys_local(self) -> List[Any]:
        if self.has_local_storage:
            return list(self._kv_store.keys())
        else:
            return []

    async def akeys(self) -> List[Any]:
        # Return keys across the cluster, not only in this process
        return await self.acall_actor_method(
            self.cluster_servlet, "aget_key_to_env_servlet_name_dict_keys"
        )

    def keys(self) -> List[Any]:
        return sync_function(self.akeys)()

    ##############################################
    # KV Store: Put
    ##############################################
    async def aput_for_env_servlet_name(
        self,
        env_servlet_name: str,
        key: Any,
        data: Any,
        serialization: Optional[str] = None,
    ):
        return await self.acall_env_servlet_method(
            env_servlet_name,
            "aput_local",
            key,
            data=data,
            serialization=serialization,
        )

    async def aput_local(self, key: Any, value: Any):
        if self.has_local_storage:
            self._kv_store[key] = value
            await self._aput_env_servlet_name_for_key(key, self.servlet_name)
        else:
            raise NoLocalObjStoreError()

    async def aput(
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
        if await self.aget(key, default=None) is not None:
            logger.warning("Key already exists in some env, overwriting.")
            await self.apop(key)

        if self.has_local_storage and env == self.servlet_name:
            if serialization is not None:
                raise ObjStoreError(
                    "We should never reach this branch if serialization is not None."
                )
            await self.aput_local(key, value)
        else:
            await self.aput_for_env_servlet_name(env, key, value, serialization)

    def put(
        self,
        key: Any,
        value: Any,
        env: Optional[str] = None,
        serialization: Optional[str] = None,
        create_env_if_not_exists: bool = False,
    ):
        return sync_function(self.aput)(
            key, value, env, serialization, create_env_if_not_exists
        )

    ##############################################
    # KV Store: Get
    ##############################################
    async def aget_from_env_servlet_name(
        self,
        env_servlet_name: str,
        key: Any,
        default: Optional[Any] = None,
        serialization: Optional[str] = None,
        remote: bool = False,
    ):
        logger.info(f"Getting {key} from servlet {env_servlet_name}")
        return await self.acall_env_servlet_method(
            env_servlet_name,
            "aget_local",
            key,
            default=default,
            serialization=serialization,  # Crucial that this is a kwarg, or the wrapper doesn't pick it up!!
            remote=remote,
        )

    def get_from_env_servlet_name(
        self,
        env_servlet_name: str,
        key: Any,
        default: Optional[Any] = None,
        serialization: Optional[str] = None,
        remote: bool = False,
    ):
        return sync_function(self.aget_from_env_servlet_name)(
            env_servlet_name, key, default, serialization, remote
        )

    def get_local(self, key: Any, default: Optional[Any] = None, remote: bool = False):
        if self.has_local_storage:
            try:
                res = self._kv_store[key]
                if remote:
                    if hasattr(res, "config"):
                        return res.config()
                    else:
                        raise ValueError(
                            f"Cannot return remote for non-Resource object of type {type(res)}."
                        )
                return res
            except KeyError as e:
                if default == KeyError:
                    raise e
                return default
        else:
            if default == KeyError:
                raise KeyError(f"No local store exists; key {key} not found.")
            return default

    async def aget(
        self,
        key: Any,
        serialization: Optional[str] = None,
        remote: bool = False,
        default: Optional[Any] = None,
    ):
        env_servlet_name_containing_key = await self.aget_env_servlet_name_for_key(key)

        if not env_servlet_name_containing_key:
            if default == KeyError:
                raise KeyError(f"No local store exists; key {key} not found.")
            return default

        if (
            env_servlet_name_containing_key == self.servlet_name
            and self.has_local_storage
        ):
            # Short-circuit route if we're already in the right env
            res = self.get_local(
                key,
                remote=remote,
                default=default,
            )
        else:
            # Note, if serialization is not None here and remote is True we won't enter the block below,
            # because the EnvServlet already packaged the config into a Response object. This is desired, as we
            # only want the remote object to be reconstructed when it's being returned to the user, which would
            # not be here if serialization is not None (probably the HTTPClient).
            res = await self.aget_from_env_servlet_name(
                env_servlet_name_containing_key,
                key,
                default=default,
                serialization=serialization,
                remote=remote,
            )

        # When the user called the obj_store.get with remote directly, we need to
        # package the config back into the remote object here before returning it.
        if (
            remote
            and (serialization is None or serialization == "none")
            and isinstance(res, dict)
            and "resource_type" in res
        ):
            config = res
            if config.get("system") == await self.aget_cluster_config():
                from runhouse import here

                config["system"] = here
            from runhouse.resources.resource import Resource

            res_copy = Resource.from_config(config=config, dryrun=True)
            return res_copy

        return res

    def get(
        self,
        key: Any,
        serialization: Optional[str] = None,
        remote: bool = False,
        default: Optional[Any] = None,
    ):
        return sync_function(self.aget)(key, serialization, remote, default)

    ##############################################
    # KV Store: Contains
    ##############################################
    async def acontains_for_env_servlet_name(self, env_servlet_name: str, key: Any):
        return await self.acall_env_servlet_method(
            env_servlet_name, "acontains_local", key
        )

    def contains_local(self, key: Any):
        if self.has_local_storage:
            return key in self._kv_store
        else:
            return False

    async def acontains(self, key: Any):
        if self.contains_local(key):
            return True

        env_servlet_name = await self.aget_env_servlet_name_for_key(key)
        if env_servlet_name == self.servlet_name and self.has_local_storage:
            raise ObjStoreError(
                "Key not found in kv store despite env servlet specifying that it is here."
            )

        if env_servlet_name is None:
            return False

        return await self.acontains_for_env_servlet_name(env_servlet_name, key)

    def contains(self, key: Any):
        return sync_function(self.acontains)(key)

    ##############################################
    # KV Store: Pop
    ##############################################
    async def apop_from_env_servlet_name(
        self,
        env_servlet_name: str,
        key: Any,
        serialization: Optional[str] = "pickle",
        *args,
    ) -> Any:
        return await self.acall_env_servlet_method(
            env_servlet_name,
            "apop_local",
            key,
            serialization,
            *args,
        )

    async def apop_local(self, key: Any, *args) -> Any:
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
            env_name = await self._apop_env_servlet_name_for_key(key, None)
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

    async def apop(
        self, key: Any, serialization: Optional[str] = "pickle", *args
    ) -> Any:
        try:
            return await self.apop_local(key)
        except KeyError as e:
            key_err = e

        # The key was not found in this env
        # So, we check the global key to env cache to see if it's elsewhere
        env_servlet_name = await self.aget_env_servlet_name_for_key(key)
        if env_servlet_name:
            if env_servlet_name == self.servlet_name and self.has_local_storage:
                raise ObjStoreError(
                    "The key was not found in this env, but the global env for key cache says it's here."
                )
            else:
                # The key was found in another env, so we need to pop it from there
                return await self.apop_from_env_servlet_name(
                    env_servlet_name, key, serialization
                )
        else:
            # Was not found in any env
            if args:
                return args[0]
            else:
                raise key_err

    def pop(self, key: Any, serialization: Optional[str] = "pickle", *args) -> Any:
        return sync_function(self.apop)(key, serialization, *args)

    ##############################################
    # KV Store: Delete
    ##############################################
    async def adelete_for_env_servlet_name(self, env_servlet_name: str, key: Any):
        return await self.acall_env_servlet_method(
            env_servlet_name, "adelete_local", key
        )

    async def adelete_local(self, key: Any):
        await self.apop_local(key)

    async def adelete_env_contents(self, env_name: Any):

        # delete the env servlet actor and remove its references
        if env_name in self.env_servlet_cache:
            actor = self.env_servlet_cache[env_name]
            ray.kill(actor)
            del self.env_servlet_cache[env_name]

        deleted_keys = await self.aclear_all_references_to_env_servlet_name(env_name)
        return deleted_keys

    def delete_env_contents(self, env_name: Any):
        return sync_function(self.adelete_env_contents)(env_name)

    async def adelete(self, key: Union[Any, List[Any]]):
        keys_to_delete = [key] if isinstance(key, str) else key
        deleted_keys = []

        for key_to_delete in keys_to_delete:
            if key_to_delete in await self.aget_all_initialized_env_servlet_names():
                deleted_keys += await self.adelete_env_contents(key_to_delete)

            if key_to_delete in deleted_keys:
                continue

            if self.contains_local(key_to_delete):
                await self.adelete_local(key_to_delete)
                deleted_keys.append(key_to_delete)
            else:
                env_servlet_name = await self.aget_env_servlet_name_for_key(
                    key_to_delete
                )
                if env_servlet_name == self.servlet_name and self.has_local_storage:
                    raise ObjStoreError(
                        "Key not found in kv store despite env servlet specifying that it is here."
                    )
                if env_servlet_name is None:
                    raise KeyError(f"Key {key} not found in any env.")

                await self.adelete_for_env_servlet_name(env_servlet_name, key_to_delete)
                deleted_keys.append(key_to_delete)

    def delete(self, key: Union[Any, List[Any]]):
        return sync_function(self.adelete)(key)

    ##############################################
    # KV Store: Clear
    ##############################################
    async def aclear_for_env_servlet_name(self, env_servlet_name: str):
        return await self.acall_env_servlet_method(env_servlet_name, "aclear_local")

    async def aclear_local(self):
        if self.has_local_storage:
            # Use asyncio gather to run all the deletes concurrently
            await asyncio.gather(
                *[self.apop_local(k) for k in list(self._kv_store.keys())]
            )

    async def aclear(self):
        logger.warning("Clearing all keys from all envs in the object store!")
        for env_servlet_name in await self.aget_all_initialized_env_servlet_names():
            if env_servlet_name == self.servlet_name and self.has_local_storage:
                await self.aclear_local()
            else:
                await self.aclear_for_env_servlet_name(env_servlet_name)

    def clear(self):
        return sync_function(self.aclear)()

    ##############################################
    # KV Store: Rename
    ##############################################
    async def arename_for_env_servlet_name(
        self, env_servlet_name: str, old_key: Any, new_key: Any
    ):
        return await self.acall_env_servlet_method(
            env_servlet_name,
            "arename_local",
            old_key,
            new_key,
        )

    async def arename_local(self, old_key: Any, new_key: Any):
        if self.servlet_name is None or not self.has_local_storage:
            raise NoLocalObjStoreError()

        obj = await self.apop(old_key)
        if obj is not None and hasattr(obj, "rns_address"):
            # Note - we set the obj.name here so the new_key is correctly turned into an rns_address, whether its
            # a full address or just a name. Then, the new_key is set to just the name so its store properly in the
            # kv store.
            obj.name = new_key  # new_key can be an rns_address! e.g. if called by Module.rename
            new_key = obj.name  # new_key is now just the name

        # By passing default, we don't throw an error if the key is not found
        await self.aput(new_key, obj, env=self.servlet_name)

    async def arename(self, old_key: Any, new_key: Any):
        # We also need to rename the resource itself
        env_servlet_name_containing_old_key = await self.aget_env_servlet_name_for_key(
            old_key
        )
        if (
            env_servlet_name_containing_old_key == self.servlet_name
            and self.has_local_storage
        ):
            await self.arename_local(old_key, new_key)
        else:
            await self.arename_for_env_servlet_name(
                env_servlet_name_containing_old_key, old_key, new_key
            )

    def rename(self, old_key: Any, new_key: Any):
        return sync_function(self.arename)(old_key, new_key)

    ##############################################
    # KV Store: Call
    ##############################################
    async def acall_for_env_servlet_name(
        self,
        env_servlet_name: str,
        key: Any,
        method_name: str,
        data: Any = None,
        serialization: Optional[str] = None,
        run_name: Optional[str] = None,
        stream_logs: bool = False,
        remote: bool = False,
    ):
        return await self.acall_env_servlet_method(
            env_servlet_name,
            "acall_local",
            key,
            method_name=method_name,
            data=data,
            serialization=serialization,
            run_name=run_name,
            stream_logs=stream_logs,
            remote=remote,
            ctx=dict(req_ctx.get()),
        )

    async def acall_local(
        self,
        key: str,
        method_name: Optional[str] = None,
        *args,
        run_name: Optional[str] = None,
        stream_logs: bool = False,
        remote: bool = False,
        **kwargs,
    ):
        """Base call functionality: Load the module, and call a method on it with args and kwargs. Nothing else.

        Handles calls on properties, methods, coroutines, and generators.

        """
        if self.servlet_name is None or not self.has_local_storage:
            raise NoLocalObjStoreError()

        log_ctx = None
        if stream_logs and run_name is not None:
            log_ctx = LogToFolder(name=run_name)
            log_ctx.__enter__()

        # Use a finally to track the active functions so that it is always removed
        request_id = req_ctx.get().request_id

        # There can be many function calls in one request_id, since request_id is tied to a call from the
        # client to the server.
        # We store with this func_call_id so we can easily pop the active call info out after the function
        # concludes. In theory we could use a tuple of (key, start_time, etc), but it doesn't accomplish much
        func_call_id = uuid.uuid4()
        self.active_function_calls[func_call_id] = ActiveFunctionCallInfo(
            key=key,
            method_name=method_name,
            request_id=request_id,
            start_time=time.time(),
        )
        try:
            res = await self._acall_local_helper(
                key,
                method_name,
                *args,
                run_name=run_name,
                stream_logs=stream_logs,
                remote=remote,
                **kwargs,
            )
        finally:
            del self.active_function_calls[func_call_id]
            if log_ctx:
                log_ctx.__exit__(None, None, None)

        return res

    async def _acall_local_helper(
        self,
        key: str,
        method_name: Optional[str] = None,
        *args,
        run_name: Optional[str] = None,
        stream_logs: bool = False,
        remote: bool = False,
        **kwargs,
    ):
        """acall_local primarily sets up the logging and tracking for the function call, then calls
        _acall_local_helper to actually do the work. This is so we can have a finally block in acall_local to clean up
        the active function calls tracking."""
        obj = self.get_local(key, default=KeyError)

        from runhouse.resources.module import Module
        from runhouse.resources.resource import Resource

        if self.is_den_auth_enabled():
            if not isinstance(obj, Resource) or obj.visibility not in [
                ResourceVisibility.UNLISTED,
            ]:
                ctx = req_ctx.get()
                if not ctx or not ctx.token:
                    raise PermissionError(
                        "No Runhouse token provided. Try running `$ runhouse login` or visiting "
                        "https://run.house/login to retrieve a token. If calling via HTTP, please "
                        "provide a valid token in the Authorization header.",
                    )

                # Setting to None in the case of non-resource or no rns_address will force auth to only
                # succeed if the user has WRITE or READ access to the cluster
                resource_uri = obj.rns_address if hasattr(obj, "rns_address") else None
                if key != self.servlet_name and not await self.ahas_resource_access(
                    ctx.token, resource_uri
                ):
                    # Do not validate access to the default Env
                    raise PermissionError(
                        f"Unauthorized access to resource {key}.",
                    )

        # Process any inputs which need to be resolved
        args = [
            arg.fetch() if (isinstance(arg, Module) and arg._resolve) else arg
            for arg in args
        ]
        kwargs = {
            k: v.fetch() if (isinstance(v, Module) and v._resolve) else v
            for k, v in kwargs.items()
        }

        method_name = method_name or "__call__"

        method_is_coroutine = False
        try:
            if isinstance(obj, Module):
                # Force this to be fully local for Modules so we don't have any circular stuff calling into other
                # envs or systems.
                method = getattr(obj.local, method_name)
                module_signature = obj.signature(rich=True)
                method_is_coroutine = (
                    method_name in module_signature
                    and module_signature[method_name]["async"]
                    and not module_signature[method_name]["gen"]
                )
            else:
                method = getattr(obj, method_name)
                method_is_coroutine = inspect.iscoroutinefunction(method)
        except AttributeError:
            logger.debug(obj.__dict__)
            raise ValueError(f"Method {method_name} not found on module {obj}")

        if method_name in ["remote_anext", "remote_await"]:
            # If the method is a coroutine or generator, we need to call it with await
            # This is a special case because when the method comes through first time around, we notice that the
            # result is a coroutine, repackage it in a "FutureModule" or "AsyncGeneratorModule", and then return
            # it to the user. Then, when the user actually awaits, they call our magic `__await__` method, which
            # calls our synchronous dunder method, which is then caught here and *actually* awaited.
            logger.info(
                f"{self.servlet_name} env: Calling special method via await {method_name} on module {key}"
            )
            res = await method(*args, **kwargs)

        elif hasattr(method, "__call__") or method_name == "__call__":
            # If method is callable, call it and return the result
            logger.info(
                f"{self.servlet_name} env: Calling method {method_name} on module {key}"
            )

            res = await arun_in_thread(method, *args, **kwargs)

            # If this was a coroutine function (not a function returning a corotuine), we need to await it here,
            # and not use the FutureModule

            # Note that for async ops, we are using the single thread within our object store for the actual
            # execution of the async function. This means if it is poorly written user code that is blocking,
            # it will block the entire object store.
            if method_is_coroutine:
                res = await res

        else:
            if args and len(args) == 1:
                # if there's an arg, this is a "set" call on the property
                logger.info(
                    f"{self.servlet_name} servlet: Setting property {method_name} on module {key}"
                )
                if isinstance(obj, Module):
                    setattr(obj.local, method_name, args[0])
                else:
                    setattr(obj, method_name, args[0])
                res = None
            else:
                # Method is a property, return the value
                logger.info(
                    f"{self.servlet_name} servlet: Getting property {method_name} on module {key}"
                )
                res = method

        laziness_type = (
            "coroutine"
            if inspect.iscoroutine(res)
            else "generator"
            if inspect.isgenerator(res)
            else "async generator"
            if inspect.isasyncgen(res)
            else None
        )

        # Make sure there's a run_name (if called through the HTTPServer there will be, but directly
        # through the ObjStore there may not be)
        run_name = run_name or generate_default_name(
            prefix=key if method_name == "__call__" else f"{key}_{method_name}",
            precision="ms",  # Higher precision because we see collisions within the same second
            sep="@",
        )

        if laziness_type:
            # If the result is a coroutine or generator, we can't return it over the process boundary
            # and need to store it to be retrieved later. In this case we return a "retrievable".
            logger.debug(
                f"{self.servlet_name} servlet: Method {method_name} on module {key} is a {laziness_type}. "
                f"Storing result to be retrieved later at result key {res}."
            )
            fut = self._construct_call_retrievable(res, run_name, laziness_type)
            await self.aput_local(run_name, fut)
            return fut

        from runhouse.resources.resource import Resource

        if isinstance(res, Resource):
            if run_name and "@" not in run_name:
                # This is a user-specified name, so we want to override the existing name with it
                # and save the resource
                res.name = run_name or res.name
                await self.aput_local(res.name, res)

            if remote:
                # If we've reached this block then we know "@" is in run_name and it's an auto-generated name,
                # so we don't want override the existing name with it (as we do above with user-specified name)
                res.name = res.name or run_name

                # Need to save the resource in case we haven't yet (e.g. if run_name was auto-generated)
                await self.aput_local(res.name, res)
                # If remote is True and the result is a resource, we return just the config
                res = res.config()

        return res

    @staticmethod
    def _construct_call_retrievable(res, res_key, laziness_type):
        if laziness_type == "coroutine":
            from runhouse.resources.future_module import FutureModule

            # TODO make this one-time-use
            return FutureModule(future=res, name=res_key)

        elif laziness_type == "generator":
            from runhouse.resources.future_module import GeneratorModule

            return GeneratorModule(future=res, name=res_key)

        elif laziness_type == "async generator":
            from runhouse.resources.future_module import AsyncGeneratorModule

            return AsyncGeneratorModule(future=res, name=res_key)

        else:
            raise ValueError(f"Invalid laziness type {laziness_type}")

    @context_wrapper
    async def acall(
        self,
        key: str,
        method_name: Optional[str] = None,
        data: Any = None,
        serialization: Optional[str] = None,
        run_name: Optional[str] = None,
        stream_logs: bool = False,
        remote: bool = False,
    ):
        env_servlet_name_containing_key = await self.aget_env_servlet_name_for_key(key)
        if not env_servlet_name_containing_key:
            raise ObjStoreError(
                f"Key {key} not found in any env, cannot call method {method_name} on it."
            )

        if (
            env_servlet_name_containing_key == self.servlet_name
            and self.has_local_storage
        ):
            from runhouse.servers.http.http_utils import deserialize_data

            deserialized_data = deserialize_data(data, serialization) or {}
            args, kwargs = deserialized_data.get("args", []), deserialized_data.get(
                "kwargs", {}
            )

            res = await self.acall_local(
                key,
                method_name,
                run_name=run_name,
                stream_logs=stream_logs,
                remote=remote,
                *args,
                **kwargs,
            )
        else:
            res = await self.acall_for_env_servlet_name(
                env_servlet_name_containing_key,
                key,
                method_name,
                data=data,
                serialization=serialization,
                run_name=run_name,
                stream_logs=stream_logs,
                remote=remote,
            )

        if remote and isinstance(res, dict) and "resource_type" in res:
            config = res
            # Config is condensed by default, so system may just be a string
            if isinstance(config.get("system"), dict):
                system_name = config.get("system").get("name")
            else:
                system_name = config.get("system")
            if system_name == (await self.aget_cluster_config()).get("name"):
                from runhouse import here

                config["system"] = here
            from runhouse.resources.resource import Resource

            res_copy = Resource.from_config(config=config, dryrun=True)
            return res_copy

        return res

    def call(
        self,
        key: str,
        method_name: Optional[str] = None,
        data: Any = None,
        serialization: Optional[str] = None,
        run_name: Optional[str] = None,
        stream_logs: bool = False,
        remote: bool = False,
    ):
        return sync_function(self.acall)(
            key,
            method_name,
            data,
            serialization,
            run_name,
            stream_logs,
            remote,
        )

    ##############################################
    # Get several keys for function initialization utilities
    ##############################################
    async def aget_list(self, keys: List[str], default: Optional[Any] = None):
        return await asyncio.gather(
            *[self.aget(key, default=default or key) for key in keys]
        )

    def get_list(self, keys: List[str], default: Optional[Any] = None):
        return sync_function(self.aget_list)(keys, default)

    async def aget_obj_refs_list(self, keys: List[Any]):

        return await asyncio.gather(
            *[
                self.aget(key, default=key) if isinstance(key, str) else key
                for key in keys
            ]
        )

    def get_obj_refs_list(self, keys: List[Any]):
        return sync_function(self.aget_obj_refs_list)(keys)

    async def aget_obj_refs_dict(self, d: Dict[Any, Any]):
        return {
            k: await self.aget(v, default=v) if isinstance(v, str) else v
            for k, v in d.items()
        }

    def get_obj_refs_dict(self, d: Dict[Any, Any]):
        return sync_function(self.aget_obj_refs_dict)(d)

    ##############################################
    # More specific helpers
    ##############################################
    async def aput_resource(
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
            resource_config, state, dryrun = tuple(
                deserialize_data(serialized_data, serialization)
            )
            return await self.aput_resource_local(resource_config, state, dryrun)

        # Normally, serialization and deserialization happens within the servlet
        # However, if we're putting an env, we need to deserialize it here and
        # actually create the corresponding env servlet.
        resource_config, _, _ = tuple(deserialize_data(serialized_data, serialization))
        if resource_config["resource_type"] == "env":
            # Note that the passed in `env_name` and the `env_name_to_create` here are
            # distinct. The `env_name` is the name of the env servlet where we want to store
            # the resource itself. The `env_name_to_create` is the name of the env servlet
            # that we need to create because we are putting an env resource somewhere on the cluster.
            runtime_env = (
                {"conda_env": resource_config["env_name"]}
                if resource_config["resource_subtype"] == "CondaEnv"
                else {}
            )

            _ = self.get_env_servlet(
                env_name=env_name,
                create=True,
                runtime_env=runtime_env,
                resources=resource_config.get("compute", None),
            )

        return await self.acall_env_servlet_method(
            env_name,
            "aput_resource_local",
            data=serialized_data,
            serialization=serialization,
        )

    def put_resource(
        self,
        serialized_data: Any,
        serialization: Optional[str] = None,
        env_name: Optional[str] = None,
    ) -> "Response":
        return sync_function(self.aput_resource)(
            serialized_data, serialization, env_name
        )

    async def aput_resource_local(
        self,
        resource_config: Dict[str, Any],
        state: Dict[Any, Any],
        dryrun: bool,
    ) -> str:
        from runhouse.resources.module import Module
        from runhouse.resources.resource import Resource

        state = state or {}
        # Resolve any sub-resources which are string references to resources already sent to this cluster.
        # We need to pop the resource's own name so it doesn't get resolved if it's already present in the
        # obj_store.
        name = resource_config.pop("name")
        subtype = resource_config.pop("resource_subtype")
        provider = (
            resource_config.pop("provider") if "provider" in resource_config else None
        )

        resource_config = await self.aget_obj_refs_dict(resource_config)
        resource_config["name"] = name
        resource_config["resource_subtype"] = subtype
        if provider:
            resource_config["provider"] = provider

        logger.debug(f"Message received from client to construct resource: {name}")

        resource = Resource.from_config(config=resource_config, dryrun=dryrun)

        for attr, val in state.items():
            setattr(resource, attr, val)

        name = resource.name or generate_default_name(prefix=resource.RESOURCE_TYPE)
        if isinstance(resource, Module):
            resource.rename(name)
        else:
            resource.name = name

        await self.aput_local(resource.name, resource)

        # Return the name in case we had to set it
        return resource.name

    ##############################################
    # Cluster info methods
    ##############################################
    def keys_with_info(self):
        if not self.has_local_storage or self.servlet_name is None:
            raise NoLocalObjStoreError()

        # Need to copy to avoid race conditions
        current_active_function_calls = copy.copy(self.active_function_calls)

        keys_and_info = {}
        for k, v in self._kv_store.items():
            cls = type(v)
            py_module = cls.__module__
            cls_name = (
                cls.__qualname__
                if py_module == "builtins"
                else (py_module + "." + cls.__qualname__)
            )

            active_fn_calls = [
                call_info.model_dump()
                for call_info in current_active_function_calls.values()
                if call_info.key == k
            ]

            keys_and_info[k] = {
                "resource_type": cls_name,
                "active_function_calls": active_fn_calls,
            }

        return keys_and_info

    async def astatus(self, send_to_den: bool = False):
        return await self.acall_actor_method(
            self.cluster_servlet, "status", send_to_den
        )

    def status(self, send_to_den: bool = False):
        return sync_function(self.astatus)(send_to_den=send_to_den)
