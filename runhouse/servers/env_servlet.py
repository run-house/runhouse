import json
import logging
import traceback
from functools import wraps
from typing import Any, Optional

from runhouse.globals import obj_store

from runhouse.resources.resource import Resource
from runhouse.rns.utils.api import ResourceVisibility

from runhouse.servers.http.http_utils import (
    deserialize_data,
    handle_exception_response,
    OutputType,
    pickle_b64,
    Response,
    serialize_data,
)
from runhouse.servers.obj_store import ClusterServletSetupOption

logger = logging.getLogger(__name__)


def error_handling_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        EnvServlet.register_activity()
        serialization = kwargs.get("serialization", None)
        if "data" in kwargs:
            serialized_data = kwargs.get("data", None)
            deserialized_data = deserialize_data(serialized_data, serialization)
            kwargs["data"] = deserialized_data

        # If serialization is None, then we have not called this from the server,
        # so we should return the result of the function directly, or raise
        # the exception if there is one, instead of returning a Response object.
        try:
            output = func(*args, **kwargs)
            if serialization is None:
                return output
            if output is not None:
                serialized_data = serialize_data(output, serialization)
                return Response(
                    output_type=OutputType.RESULT_SERIALIZED,
                    data=serialized_data,
                    serialization=serialization,
                )
            else:
                return Response(
                    output_type=OutputType.SUCCESS,
                )
        except Exception as e:
            return handle_exception_response(e, traceback.format_exc(), serialization)

    return wrapper


class EnvServlet:
    def __init__(self, env_name: str, *args, **kwargs):
        self.env_name = env_name

        obj_store.initialize(
            self.env_name,
            has_local_storage=True,
            setup_cluster_servlet=ClusterServletSetupOption.GET_OR_FAIL,
        )

        self.output_types = {}
        self.thread_ids = {}

    @staticmethod
    def register_activity():
        try:
            from sky.skylet.autostop_lib import set_last_active_time_to_now

            set_last_active_time_to_now()
        except ImportError:
            pass

    ##############################################################
    # Methods decorated with a standardized error decorating handler
    # These catch exceptions and wrap the output in a Response object.
    # They also handle arbitrary serialization and deserialization.
    # NOTE: These need to take in "data" and "serialization" as arguments
    # even if unused, because they are used by the decorator
    ##############################################################
    @error_handling_decorator
    def put_resource_local(
        self,
        data: Any,  # This first comes in a serialized format which the decorator re-populates after deserializing
        serialization: Optional[str] = None,
    ):
        resource_config, state, dryrun = data
        return obj_store.put_resource_local(resource_config, state, dryrun)

    @error_handling_decorator
    def put_local(self, key: Any, data: Any, serialization: Optional[str] = None):
        return obj_store.put_local(key, data)

    @error_handling_decorator
    def call_local(
        self,
        key: Any,
        method_name: str = None,
        data: Any = None,
        serialization: Optional[str] = None,
        run_name: Optional[str] = None,
        stream_logs: bool = False,
        remote: bool = False,
    ):
        args, kwargs = data or ([], {})
        return obj_store.call_local(
            key,
            method_name,
            run_name=run_name,
            stream_logs=stream_logs,
            remote=remote,
            *args,
            **kwargs,
        )

    @error_handling_decorator
    def get_local(
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
    def keys_local(self):
        self.register_activity()
        return obj_store.keys_local()

    def rename_local(self, key: Any, new_key: Any):
        self.register_activity()
        return obj_store.rename_local(key, new_key)

    def contains_local(self, key: Any):
        self.register_activity()
        return obj_store.contains_local(key)

    def pop_local(self, key: Any, *args):
        self.register_activity()
        return obj_store.pop_local(key, *args)

    def delete_local(self, key: Any):
        self.register_activity()
        return obj_store.delete_local(key)

    def clear_local(self):
        self.register_activity()
        return obj_store.clear_local()

    def call(
        self,
        module_name: str,
        method=None,
        args=None,
        kwargs=None,
        serialization="json",
        token_hash=None,
        den_auth=False,
    ):
        self.register_activity()
        module = obj_store.get(module_name, default=KeyError)
        if den_auth:
            if not isinstance(module, Resource) or module.visibility not in [
                ResourceVisibility.UNLISTED,
                ResourceVisibility.PUBLIC,
                "unlisted",
                "public",
            ]:
                resource_uri = (
                    module.rns_address if hasattr(module, "rns_address") else None
                )
                if not obj_store.has_resource_access(token_hash, resource_uri):
                    raise PermissionError(
                        f"No read or write access to requested resource {resource_uri}"
                    )

        if method:
            fn = getattr(module, method)
            result = fn(*(args or []), **(kwargs or {}))
        else:
            result = module

        logger.info(f"Got result back from call: {result}")

        if serialization == "json":
            return json.dumps(result)
        elif serialization == "pickle":
            return pickle_b64(result)
        elif serialization == "None":
            # e.g. if the user wants to handle their own serialization
            return result
        else:
            raise ValueError(f"Unknown serialization: {serialization}")
