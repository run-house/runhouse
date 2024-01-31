import asyncio
import inspect
import json
import logging
import threading
import time
import traceback
from functools import wraps
from typing import Any, Optional

from runhouse.globals import obj_store

from runhouse.resources.blobs import Blob
from runhouse.resources.module import Module
from runhouse.resources.provenance import run
from runhouse.resources.queues import Queue
from runhouse.resources.resource import Resource
from runhouse.rns.utils.api import ResourceVisibility

# from runhouse.rns.utils.names import _generate_default_name
from runhouse.servers.http.http_utils import (
    b64_unpickle,
    deserialize_data,
    handle_exception_response,
    Message,
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
            if serialization is None:
                raise e
            else:
                # For now, this is always "pickle" because we don't support json serialization of exceptions
                return handle_exception_response(e, traceback.format_exc())

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

    def call_module_method(
        self,
        module_name,
        method_name,
        message: Message,
        token_hash: str,
        den_auth: bool,
        serialization: Optional[str] = None,
    ):
        self.register_activity()
        result_resource = None

        persist = message.save or message.remote or message.run_async
        try:
            logger.info(
                f"Message received from client to call method {method_name} on module {module_name} at {time.time()}"
            )

            result_resource = Queue(name=message.key, persist=persist)
            result_resource.provenance = run(
                name=message.key,
                log_dest="file" if message.stream_logs else None,
                load=False,
            )
            result_resource.provenance.__enter__()

            module = obj_store.get(module_name, default=KeyError)
            if den_auth:
                if not isinstance(module, Resource) or module.visibility not in [
                    ResourceVisibility.UNLISTED,
                    ResourceVisibility.PUBLIC,
                    "unlisted",
                    "public",
                ]:
                    # Setting to None in the case of non-resource or no rns_address will force auth to only
                    # succeed if the user has WRITE or READ access to the cluster
                    resource_uri = (
                        module.rns_address if hasattr(module, "rns_address") else None
                    )
                    if not obj_store.has_resource_access(token_hash, resource_uri):
                        raise PermissionError(
                            f"No read or write access to requested resource {resource_uri}"
                        )

            self.thread_ids[
                message.key
            ] = threading.get_ident()  # Save thread ID for this message
            # Remove output types from previous runs
            self.output_types.pop(message.key, None)

            # Save now so status and initial streamed results are available globally
            if message.save:
                result_resource.save()

            # If method_name is None, return the module itself as this is a "get" request
            try:
                method = getattr(module, method_name) if method_name else module
            except AttributeError:
                logger.debug(module.__dict__)
                raise ValueError(
                    f"Method {method_name} not found on module {module_name}"
                )

            # Don't call the method if it's a property or a "get" request (returning the module itself)
            if hasattr(method, "__call__") and method_name:
                # If method is callable, call it and return the result
                logger.info(
                    f"{self.env_name} servlet: Calling method {method_name} on module {module_name}"
                )
                callable_method = True

            else:
                # Method is a property, return the value
                logger.info(
                    f"{self.env_name} servlet: Getting property {method_name} on module {module_name}"
                )
                callable_method = False

            # FastAPI automatically deserializes json
            args, kwargs = (
                b64_unpickle(message.data)
                if message.data and not serialization
                else ([], message.data)
                if serialization == "json"
                else ([], {})
            )
            # Resolve any resources which need to be resolved
            args = [
                arg.fetch() if (isinstance(arg, Module) and arg._resolve) else arg
                for arg in args
            ]
            kwargs = {
                k: v.fetch() if (isinstance(v, Module) and v._resolve) else v
                for k, v in kwargs.items()
            }

            if not callable_method and kwargs and "new_value" in kwargs:
                # If new_value was passed, that means we're setting a property
                setattr(module, method_name, kwargs["new_value"])
                result_resource.pin()
                self.output_types[message.key] = OutputType.SUCCESS
                result_resource.provenance.__exit__(None, None, None)
                return Response(output_type=OutputType.SUCCESS)

            if persist or message.stream_logs:
                result_resource.pin()

            # If method is a property, `method = getattr(module, method_name, None)` above already
            # got our result
            if inspect.iscoroutinefunction(method):
                # If method is a coroutine, we need to await it
                logger.debug(
                    f"{self.env_name} servlet: Method {method_name} on module {module_name} is a coroutine"
                )
                result = asyncio.run(method(*args, **kwargs))
            else:
                result = method(*args, **kwargs) if callable_method else method

            # TODO do we need the branch above if we do this?
            if inspect.iscoroutine(result):
                result = asyncio.run(result)

            if inspect.isgenerator(result) or inspect.isasyncgen(result):
                result_resource.pin()
                # Stream back the results of the generator
                logger.info(
                    f"Streaming back results of generator {module_name}.{method_name}"
                )
                self.output_types[message.key] = OutputType.RESULT_STREAM
                if inspect.isasyncgen(result):
                    while True:
                        try:
                            self.register_activity()
                            result_resource.put(asyncio.run(result.__anext__()))
                        except StopAsyncIteration:
                            break
                else:
                    for val in result:
                        self.register_activity()
                        # Doing this at the top of the loop so we can catch the final result and change the OutputType
                        result_resource.put(val)

                # Set run status to COMPLETED to indicate end of stream
                result_resource.provenance.__exit__(None, None, None)

                # Resave with new status
                if message.save:
                    result_resource.save()
            else:
                # If the user needs this result again later, don't put it in queue or
                # it will be gone after the first get
                if persist:
                    if isinstance(result, Resource):
                        # If the user's method returned a resource, save that resource as the result
                        # instead of the queue so it's available for global caching
                        result.provenance = result_resource.provenance
                        result.name = message.key
                        result_resource = result
                    else:
                        # We shouldn't return a queue if the result is not a generator, so replace it with a blob
                        result_resource = Blob(
                            name=message.key, provenance=result_resource.provenance
                        )
                        result_resource.data = result

                    result_resource.pin()

                    # Write out the new result_resource to the obj_store
                    # obj_store.put(message.key, result_resource, env=self.env_name)
                else:
                    if not message.stream_logs:
                        # If we don't need to persist the result or stream logs,
                        # we can return the result to the user immediately
                        result_resource.provenance.__exit__(None, None, None)
                        return Response(
                            data=pickle_b64(result)
                            if not serialization == "json"
                            else result,
                            output_type=OutputType.RESULT,
                        )
                    # Put the result in the queue so we can retrieve it once
                    result_resource.put(result)

                # If not a generator, the method was already called above and completed
                self.output_types[message.key] = OutputType.RESULT
                result_resource.provenance.__exit__(None, None, None)

                if message.save:
                    result_resource.save()
                self.register_activity()
        except Exception as e:
            logger.exception(e)
            self.register_activity()

            # Setting this here is great because it allows us to still return all the computed values of a
            # generator before hitting the exception, stream the logs back to the client until raising the exception,
            # and indicate that we hit an exception before any results are available if that's the case.
            self.output_types[message.key] = OutputType.EXCEPTION
            result_resource.pin()
            result_resource.provenance.__exit__(
                type(e), e, traceback.format_exc()
            )  # TODO use format_tb instead?

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
