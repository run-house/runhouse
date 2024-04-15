import logging
import os
import traceback
from functools import wraps
from typing import Any, Dict, Optional

from runhouse.globals import obj_store

from runhouse.servers.http.http_utils import (
    deserialize_data,
    handle_exception_response,
    OutputType,
    Response,
    serialize_data,
)
from runhouse.servers.obj_store import ClusterServletSetupOption

logger = logging.getLogger(__name__)


def error_handling_decorator(func):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        ctx = kwargs.pop("ctx", None)
        ctx_token = None
        if ctx:
            ctx_token = obj_store.set_ctx(**ctx)

        serialization = kwargs.get("serialization", None)
        if "data" in kwargs:
            serialized_data = kwargs.get("data", None)
            deserialized_data = deserialize_data(serialized_data, serialization)
            kwargs["data"] = deserialized_data

        # If serialization is None, then we have not called this from the server,
        # so we should return the result of the function directly, or raise
        # the exception if there is one, instead of returning a Response object.
        try:
            output = await func(*args, **kwargs)
            if serialization is None or serialization == "none":
                obj_store.unset_ctx(ctx_token) if ctx_token else None
                return output
            if output is not None:
                serialized_data = serialize_data(output, serialization)
                obj_store.unset_ctx(ctx_token) if ctx_token else None
                return Response(
                    output_type=OutputType.RESULT_SERIALIZED,
                    data=serialized_data,
                    serialization=serialization,
                )
            else:
                obj_store.unset_ctx(ctx_token) if ctx_token else None
                return Response(
                    output_type=OutputType.SUCCESS,
                )
        except Exception as e:
            obj_store.unset_ctx(ctx_token) if ctx_token else None
            return handle_exception_response(e, traceback.format_exc(), serialization)

    return wrapper


class EnvServlet:
    async def __init__(self, env_name: str, *args, **kwargs):
        self.env_name = env_name

        await obj_store.ainitialize(
            self.env_name,
            has_local_storage=True,
            setup_cluster_servlet=ClusterServletSetupOption.GET_OR_FAIL,
        )

        # Ray defaults to setting OMP_NUM_THREADS to 1, which unexpectedly limit parallelism in user programs.
        # We delete it by default, but if we find that the user explicitly set it to another value, we respect that.
        # This is really only a factor if the user set the value inside the VM or container, or inside the base_env
        # which a cluster was initialized with. If they set it inside the env constructor and the env was sent to the
        # cluster normally with .to, it will be set after this point.
        # TODO this had no effect when we did it below where we set CUDA_VISIBLE_DEVICES, so we may need to move that
        #  here and mirror the same behavior (setting it based on the detected gpus in the whole cluster may not work
        #  for multinode, but popping it may also break things, it needs to be tested).
        num_threads = os.environ.get("OMP_NUM_THREADS")
        if num_threads is not None and num_threads != "1":
            os.environ["OMP_NUM_THREADS"] = num_threads
        else:
            os.environ["OMP_NUM_THREADS"] = ""

        self.output_types = {}
        self.thread_ids = {}

    ##############################################################
    # Methods to disable or enable den auth
    ##############################################################
    async def aset_cluster_config(self, cluster_config: Dict[str, Any]):
        obj_store.cluster_config = cluster_config

    async def aset_cluster_config_value(self, key: str, value: Any):
        obj_store.cluster_config[key] = value

    ##############################################################
    # Methods decorated with a standardized error decorating handler
    # These catch exceptions and wrap the output in a Response object.
    # They also handle arbitrary serialization and deserialization.
    # NOTE: These need to take in "data" and "serialization" as arguments
    # even if unused, because they are used by the decorator
    ##############################################################
    @error_handling_decorator
    async def aput_resource_local(
        self,
        data: Any,  # This first comes in a serialized format which the decorator re-populates after deserializing
        serialization: Optional[str] = None,
    ):
        resource_config, state, dryrun = tuple(data)
        return await obj_store.aput_resource_local(resource_config, state, dryrun)

    @error_handling_decorator
    async def aput_local(
        self, key: Any, data: Any, serialization: Optional[str] = None
    ):
        return await obj_store.aput_local(key, data)

    @error_handling_decorator
    async def acall_local(
        self,
        key: Any,
        method_name: str = None,
        data: Any = None,
        serialization: Optional[str] = None,
        run_name: Optional[str] = None,
        stream_logs: bool = False,
        remote: bool = False,
        ctx: Optional[dict] = None,
    ):
        if data is not None:
            args, kwargs = data.get("args", []), data.get("kwargs", {})
        else:
            args, kwargs = [], {}

        return await obj_store.acall_local(
            key,
            method_name,
            run_name=run_name,
            stream_logs=stream_logs,
            remote=remote,
            *args,
            **kwargs,
        )

    @error_handling_decorator
    async def aget_local(
        self,
        key: Any,
        default: Optional[Any] = None,
        serialization: Optional[str] = None,
        remote: bool = False,
    ):
        return obj_store.get_local(key, default=default, remote=remote)

    ##############################################################
    # IPC methods for interacting with local object store only
    # These do not catch exceptions, and do not wrap the output
    # in a Response object.
    ##############################################################
    async def akeys_local(self):
        return obj_store.keys_local()

    async def arename_local(self, key: Any, new_key: Any):
        return await obj_store.arename_local(key, new_key)

    async def acontains_local(self, key: Any):
        return obj_store.contains_local(key)

    async def apop_local(self, key: Any, *args):
        return await obj_store.apop_local(key, *args)

    async def adelete_local(self, key: Any):
        return await obj_store.adelete_local(key)

    async def aclear_local(self):
        return await obj_store.aclear_local()

    async def astatus_local(self):
        return obj_store.status_local()
