import logging
import sys
import traceback
from concurrent import futures
from pathlib import Path

import grpc

import ray
import ray.cloudpickle as pickle
from sky.skylet.autostop_lib import set_last_active_time_to_now

import runhouse.grpc_handler.unary_pb2 as pb2

import runhouse.grpc_handler.unary_pb2_grpc as pb2_grpc
from runhouse.grpc_handler.unary_client import OutputType
from runhouse.rh_config import obj_store
from runhouse.rns.packages.package import Package
from runhouse.rns.run_module_utils import call_fn_by_type, get_fn_by_name
from runhouse.rns.top_level_rns_fns import (
    clear_pinned_memory,
    pinned_keys,
    remove_pinned_object,
)

logger = logging.getLogger(__name__)


class UnaryService(pb2_grpc.UnaryServicer):
    DEFAULT_PORT = 50052
    MAX_MESSAGE_LENGTH = 1 * 1024 * 1024 * 1024  # 1 GB
    LOGGING_WAIT_TIME = 1.0

    def __init__(self, *args, **kwargs):
        ray.init(address="auto")
        self._installed_packages = []
        self.register_activity()

    def register_activity(self):
        set_last_active_time_to_now()

    def InstallPackages(self, request, context):
        self.register_activity()
        # logger.info(f"Message received from client: {request.message}")
        packages = pickle.loads(request.message)
        logger.info(f"Message received from client to install packages: {packages}")
        for package in packages:
            if isinstance(package, str):
                pkg = Package.from_string(package)
            elif hasattr(package, "install"):
                pkg = package
            else:
                raise ValueError(f"package {package} not recognized")

            if (str(pkg)) in self._installed_packages:
                continue
            logger.info(f"Installing package: {str(pkg)}")
            pkg.install()
            self._installed_packages.append(str(pkg))

        self.register_activity()
        return pb2.MessageResponse(message=pickle.dumps(True), received=True)

    def GetObject(self, request, context):
        self.register_activity()
        key, stream_logs = pickle.loads(request.message)
        logger.info(f"Message received from client to get object: {key}")
        if not stream_logs:
            return pb2.MessageResponse(
                message=pickle.dumps(obj_store.get(key)),
                received=True,
                output_type=OutputType.RESULT,
            )

        logfiles = None
        open_files = None
        ret_obj = None
        returned = False
        while not returned:
            try:
                res = obj_store.get(key, timeout=self.LOGGING_WAIT_TIME)
                logger.info(f"Got object of type {type(res)} back from object store")
                ret_obj = [res, None, None]
                returned = True
                # Don't return yet, go through the loop once more to get any remaining log lines
            except ray.exceptions.GetTimeoutError:
                pass
            except ray.exceptions.TaskCancelledError as e:
                logger.info(f"Attempted to get task {key} that was cancelled.")
                returned = True
                ret_obj = [None, e, traceback.format_exc()]

            if not logfiles:
                logfiles = obj_store.get_logfiles(key)
                open_files = [open(i, "r") for i in logfiles]
                logger.info(f"Streaming logs for {key} from {logfiles}")

            # Grab all the lines written to all the log files since the last time we checked
            ret_lines = []
            for i, f in enumerate(open_files):
                file_lines = f.readlines()
                if file_lines:
                    # TODO [DG] handle .out vs .err, and multiple workers
                    # if len(logfiles) > 1:
                    #     ret_lines.append(f"Process {i}:")
                    ret_lines += file_lines
            if ret_lines:
                yield pb2.MessageResponse(
                    message=pickle.dumps(ret_lines),
                    received=True,
                    output_type=OutputType.STDOUT,
                )

        # We got the object back from the object store, so we're done (but we went through the loop once
        # more to get any remaining log lines)
        [f.close() for f in open_files]
        yield pb2.MessageResponse(
            message=pickle.dumps(ret_obj), received=True, output_type=OutputType.RESULT
        )

    def ClearPins(self, request, context):
        self.register_activity()
        pins_to_clear = pickle.loads(request.message)
        logger.info(
            f"Message received from client to clear pins: {pins_to_clear or 'all'}"
        )
        cleared = []
        if pins_to_clear:
            for pin in pins_to_clear:
                remove_pinned_object(pin)
                cleared.append(pin)
        else:
            cleared = list(pinned_keys())
            clear_pinned_memory()

        self.register_activity()
        return pb2.MessageResponse(message=pickle.dumps(cleared), received=True)

    def CancelRun(self, request, context):
        self.register_activity()
        run_keys, force = pickle.loads(request.message)
        if not hasattr(run_keys, "len"):
            run_keys = [run_keys]
        obj_refs = obj_store.get_obj_refs_list(run_keys)
        [
            ray.cancel(obj_ref, force=force, recursive=True)
            for obj_ref in obj_refs
            if isinstance(obj_ref, ray.ObjectRef)
        ]
        return pb2.MessageResponse(
            message=pickle.dumps("Cancelled"),
            received=True,
            output_type=OutputType.RESULT,
        )

    def RunModule(self, request, context):
        self.register_activity()
        # get the function result from the incoming request
        try:
            [relative_path, module_name, fn_name, fn_type, args, kwargs] = pickle.loads(
                request.message
            )

            module_path = None
            # If relative_path is None, the module is not in the working dir, and should be in the reqs
            if relative_path:
                module_path = str((Path.home() / relative_path).resolve())
                sys.path.append(module_path)
                logger.info(f"Appending {module_path} to sys.path")

            if module_name == "notebook":
                fn = fn_name  # Already unpickled above
            else:
                fn = get_fn_by_name(module_name, fn_name)

            res = call_fn_by_type(fn, fn_type, fn_name, module_path, args, kwargs)
            # [res, None, None] is a silly hack for packaging result alongside exception and traceback
            result = {"message": pickle.dumps([res, None, None]), "received": True}

            self.register_activity()
            return pb2.MessageResponse(**result)
        except Exception as e:
            logger.exception(e)
            message = [None, e, traceback.format_exc()]
            self.register_activity()
            return pb2.MessageResponse(message=pickle.dumps(message), received=False)


def serve():
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=10),
        options=[
            ("grpc.max_send_message_length", UnaryService.MAX_MESSAGE_LENGTH),
            ("grpc.max_receive_message_length", UnaryService.MAX_MESSAGE_LENGTH),
        ],
    )
    pb2_grpc.add_UnaryServicer_to_server(UnaryService(), server)
    server.add_insecure_port("[::]:50052")
    server.start()
    logger.info("Server up and running")
    server.wait_for_termination()


if __name__ == "__main__":
    serve()
