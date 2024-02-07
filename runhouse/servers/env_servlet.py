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

from runhouse.resources.blobs import blob, Blob
from runhouse.resources.module import Module
from runhouse.resources.provenance import run, RunStatus
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
            if output is not None:
                if serialization is None:
                    return output
                else:
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

    def get(
        self,
        key,
        remote=False,
        stream=False,
        timeout=None,
        serialization=None,
        _intra_cluster=False,
    ):
        """Get an object from the servlet's object store.

        Args:
            key (str): The key of the object to get.
            remote (bool): Whether to return the object or it's config to construct a remote object.
            stream (bool): Whether to stream results as available (if the key points to a queue).
        """
        self.register_activity()
        try:
            ret_obj = obj_store.get(
                key, default=KeyError, check_other_envs=not _intra_cluster
            )
            logger.debug(
                f"Servlet {self.env_name} got object of type "
                f"{type(ret_obj)} back from object store for key {key}"
            )
            if _intra_cluster:
                if remote:
                    return ret_obj.config_for_rns
                return ret_obj

            # If the request doesn't want a stream, we can just return the queue object in same way as any other, below
            if isinstance(ret_obj, Queue) and stream:
                if remote and self.output_types.get(key) in [
                    OutputType.RESULT_STREAM,
                    OutputType.SUCCESS_STREAM,
                ]:
                    # If this is a "remote" request and we already know the output type is a stream, we can
                    # return the Queue as a remote immediately so the client can start streaming the results
                    res = ret_obj.config_for_rns
                    res["dryrun"] = True
                    return Response(
                        data=res,
                        output_type=OutputType.CONFIG,
                    )

                # If we're waiting for a result, this will block until one is available, which will either
                # cause the server's ray.get to timeout so it can try again, or return None as soon as the result is
                # available so the server can try requesting again now that it's ready.
                if ret_obj.empty():
                    if (
                        not ret_obj.provenance
                        or ret_obj.provenance.status == RunStatus.NOT_STARTED
                    ):
                        while key not in self.output_types:
                            time.sleep(0.1)
                        return

                    # This allows us to return the results of a generator as they become available, rather than
                    # waiting a full second for the ray.get in the server to timeout before trying again.
                    if ret_obj.provenance.status == RunStatus.RUNNING:
                        time.sleep(0.1)
                        return

                    if ret_obj.provenance.status == RunStatus.COMPLETED:
                        # We need to look up the output type because this could be a stream with no results left,
                        # which should still return OutputType.RESULT_STREAM, or a call with no result, which should
                        # return OutputType.SUCCESS
                        if self.output_types[key] == OutputType.RESULT_STREAM:
                            return Response(output_type=OutputType.SUCCESS_STREAM)
                        else:
                            return Response(output_type=OutputType.SUCCESS)

                    if ret_obj.provenance.status == RunStatus.ERROR:
                        return Response(
                            error=pickle_b64(ret_obj.provenance.error)
                            if not serialization == "json"
                            else str(ret_obj.provenance.error),
                            traceback=pickle_b64(ret_obj.provenance.traceback)
                            if not serialization == "json"
                            else str(ret_obj.provenance.traceback),
                            output_type=OutputType.EXCEPTION,
                        )

                    if ret_obj.provenance.status == RunStatus.CANCELLED:
                        return Response(output_type=OutputType.CANCELLED)

                res = ret_obj.get(block=True, timeout=timeout)
                # There's no OutputType.EXCEPTION case to handle here, because if an exception were thrown the
                # provenance.status would be RunStatus.ERROR, and we want to continue retrieving results until the
                # queue is empty, and then will return the exception and traceback in the empty case above.
                return Response(
                    data=pickle_b64(res) if not serialization == "json" else res,
                    output_type=self.output_types[key],
                )

            # If the user requests a remote object, we can return a queue before results complete so they can
            # stream in results directly from the queue. For all other cases, we need to wait for the results
            # to be available.
            if remote:
                if not isinstance(ret_obj, Resource):
                    # If the user requests a remote of an object that is not a Resource, we need to wrap it
                    # in a Resource first, which will overwrite the original object in the object store. We
                    # may want to just throw an error instead, but let's see if this is acceptable to start.
                    # TODO just put it in the obj store and return a string instead?
                    ret_obj = blob(data=ret_obj, name=key)
                    ret_obj.pin()

                if ret_obj.provenance and ret_obj.provenance.status == RunStatus.ERROR:
                    return Response(
                        error=pickle_b64(ret_obj.provenance.error)
                        if not serialization == "json"
                        else str(ret_obj.provenance.error),
                        traceback=pickle_b64(ret_obj.provenance.traceback)
                        if not serialization == "json"
                        else str(ret_obj.provenance.traceback),
                        output_type=OutputType.EXCEPTION,
                    )

                # If this is a "remote" request, just return the rns config and the client will reconstruct the
                # resource from it
                res = ret_obj.config_for_rns
                res["dryrun"] = True
                return Response(
                    data=res,
                    output_type=OutputType.CONFIG,
                )

            if isinstance(ret_obj, Resource) and ret_obj.provenance:
                if ret_obj.provenance.status == RunStatus.ERROR:
                    return Response(
                        error=pickle_b64(ret_obj.provenance.error)
                        if not serialization == "json"
                        else str(ret_obj.provenance.error),
                        traceback=pickle_b64(ret_obj.provenance.traceback)
                        if not serialization == "json"
                        else str(ret_obj.provenance.traceback),
                        output_type=OutputType.EXCEPTION,
                    )
                # Includes the case where the user called a method with remote or save, where even if the original
                # return value wasn't a resource, we want to return the wrapped resource anyway. If the user called
                # a non-generator method without remote or save, the result would be in a queue and handled above,
                # so it'll still be returned unwrapped.
                if ret_obj.provenance.status == RunStatus.COMPLETED:
                    return Response(
                        data=pickle_b64(ret_obj)
                        if not serialization == "json"
                        else ret_obj,
                        output_type=OutputType.RESULT,
                    )

                if ret_obj.provenance.status == RunStatus.CANCELLED:
                    return Response(output_type=OutputType.CANCELLED)

                # We don't need to handle the ret_obj.provenance.status == RunStatus.NOT_STARTED case, because
                # if the run hasn't started yet, the result_resource will still be a Queue and handled above.
                # If the run has started, but for some reason the Queue hasn't been created yet (even though it's
                # created immediately), the ret_obj wouldn't be found in the obj_store.

            return Response(
                data=pickle_b64(ret_obj) if not serialization == "json" else ret_obj,
                output_type=OutputType.RESULT,
            )
        except Exception as e:
            if _intra_cluster:
                raise e

            return Response(
                error=pickle_b64(e) if not serialization == "json" else str(e),
                traceback=pickle_b64(traceback.format_exc())
                if not serialization == "json"
                else str(traceback.format_exc()),
                output_type=OutputType.EXCEPTION,
            )

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

    ##############################################################
    # IPC methods for interacting with local object store only
    # These do not catch exceptions, and do not wrap the output
    # in a Response object.
    ##############################################################
    def keys_local(self):
        self.register_activity()
        return obj_store.keys_local()

    def get_local(self, key: Any, default: Optional[Any] = None):
        self.register_activity()
        return obj_store.get_local(key, default)

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
