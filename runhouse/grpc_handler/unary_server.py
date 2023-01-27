import importlib
import sys
import logging
from pathlib import Path
import grpc
from concurrent import futures
import traceback
import os
from datetime import datetime
import yaml
from functools import wraps

import ray
import ray.cloudpickle as pickle
from runhouse.grpc_handler.unary_client import OutputType
from sky.skylet.autostop_lib import set_last_active_time_to_now

import runhouse.grpc_handler.unary_pb2_grpc as pb2_grpc
import runhouse.grpc_handler.unary_pb2 as pb2
from runhouse.rns.packages.package import Package
from runhouse.rns.top_level_rns_fns import remove_pinned_object, clear_pinned_memory, pinned_keys
from runhouse.rh_config import obj_store

logger = logging.getLogger(__name__)

RAY_LOGFILE_PATH = '/tmp/ray/session_latest/logs'


class UnaryService(pb2_grpc.UnaryServicer):
    DEFAULT_PORT = 50052
    MAX_MESSAGE_LENGTH = 1 * 1024 * 1024 * 1024  # 1 GB
    LOGGING_WAIT_TIME = 1.0

    def __init__(self, *args, **kwargs):
        ray.init(address="auto")

        self.sky_data = {}
        with open(os.path.expanduser("~/.sky/sky_ray.yml")) as f:
            self.sky_data = yaml.safe_load(f)
        self.cluster_name = self.sky_data.get('cluster_name', 'cluster')

        self.imported_modules = {}
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
            elif hasattr(package, 'install'):
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
            return pb2.MessageResponse(message=pickle.dumps(obj_store.get(key)),
                                       received=True,
                                       output_type=OutputType.RESULT)

        logfiles = None
        open_files = None
        ret_obj = None
        returned = False
        while not returned:
            try:
                ret_obj = obj_store.get(key, timeout=self.LOGGING_WAIT_TIME)
                logger.info(f"Got object of type {type(ret_obj)} back from object store")
                returned = True
                # Don't return yet, go through the loop once more to get any remaining log lines
            except ray.exceptions.GetTimeoutError:
                pass

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
                yield pb2.MessageResponse(message=pickle.dumps(ret_lines),
                                          received=True, output_type=OutputType.STDOUT)

        # We got the object back from the object store, so we're done (but we went through the loop once
        # more to get any remaining log lines)
        [f.close() for f in open_files]
        yield pb2.MessageResponse(message=pickle.dumps(ret_obj),
                                  received=True,
                                  output_type=OutputType.RESULT)

    def ClearPins(self, request, context):
        pins_to_clear = pickle.loads(request.message)
        logger.info(f"Message received from client to clear pins: {pins_to_clear or 'all'}")
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

    def GetServerResponse(self, request, context):

        self.register_activity()
        # get the function result from the incoming request
        try:
            [relative_path, module_name, fn_name, fn_type, args, kwargs] = pickle.loads(request.message)

            module_path = None
            # If relative_path is None, the module is not in the working dir, and should be in the reqs
            if relative_path:
                module_path = str((Path.home() / relative_path).resolve())
                sys.path.append(module_path)
                logger.info(f"Appending {module_path} to sys.path")

            if module_name == "notebook":
                fn = fn_name  # Already unpickled above
            else:
                if module_name in self.imported_modules:
                    importlib.invalidate_caches()
                    self.imported_modules[module_name] = importlib.reload(self.imported_modules[module_name])
                    logger.info(f"Reloaded module {module_name}")
                else:
                    self.imported_modules[module_name] = importlib.import_module(module_name)
                    logger.info(f"Importing module {module_name}")
                fn = getattr(self.imported_modules[module_name], fn_name)

            res = call_fn_on_cluster(fn, fn_type, fn_name, module_path, args, kwargs)
            # [res, None, None] is a silly hack for packaging result alongside exception and traceback
            result = {'message': pickle.dumps([res, None, None]), 'received': True}

            self.register_activity()
            return pb2.MessageResponse(**result)
        except Exception as e:
            logger.exception(e)
            message = [None, e, traceback.format_exc()]
            self.register_activity()
            return pb2.MessageResponse(message=pickle.dumps(message), received=False)


def enable_logging_fn_wrapper(fn, run_key):
    @wraps(fn)
    def wrapped_fn(*inner_args, **inner_kwargs):
        """Sets the logfiles for a function to be the logfiles for the current ray worker.
        This is used to stream logs from the worker to the client."""
        worker_id = ray._private.worker.global_worker.worker_id.hex()
        logdir = Path.home() / ".rh/logs" / run_key
        logdir.mkdir(exist_ok=True, parents=True)
        ray_logs_path = Path(RAY_LOGFILE_PATH)
        stdout_files = list(ray_logs_path.glob(f'worker-{worker_id}-*'))
        # Create simlinks to the ray log files in the run directory
        logger.info(f"Writing logs on cluster to {logdir}")
        for f in stdout_files:
            symlink = logdir / f.name
            if not symlink.exists():
                symlink.symlink_to(f)
        return fn(*inner_args, **inner_kwargs)

    return wrapped_fn


def call_fn_on_cluster(fn, fn_type, fn_name, module_path, args, kwargs):
    run_key = f"{fn_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # TODO other possible fn_types: 'batch', 'streaming'
    if fn_type == 'call':
        args = [obj_store.get(arg, default=arg) if isinstance(arg, str) else arg for arg in args]
        kwargs = {k: obj_store.get(v, default=v) if isinstance(v, str) else v for k, v in kwargs.items()}
        res = fn(*args, **kwargs)
    elif fn_type == 'get':
        obj_ref = args[0]
        res = ray.get(obj_ref)
    else:
        args = obj_store.get_obj_refs_list(args)
        kwargs = obj_store.get_obj_refs_dict(kwargs)

        logging_wrapped_fn = enable_logging_fn_wrapper(fn, run_key)

        # We need to add the module_path to the PYTHONPATH because ray runs remotes in a new process
        # We need to set max_calls to make sure ray doesn't cache the remote function and ignore changes to the module
        # See: https://docs.ray.io/en/releases-2.2.0/ray-core/package-ref.html#ray-remote
        # We need non-zero cpus and gpus for Ray to allow access to the compute.
        # We should see if there's a more elegant way to specify this.
        ray_fn = ray.remote(num_cpus=0.0001, num_gpus=0.0001,
                            max_calls=len(args) if fn_type in ['map', 'starmap'] else 1,
                            runtime_env={"env_vars": {"PYTHONPATH": module_path or ''}})(logging_wrapped_fn)
        if fn_type == 'map':
            obj_ref = [ray_fn.remote(arg, **kwargs) for arg in args]
        elif fn_type == 'starmap':
            obj_ref = [ray_fn.remote(*arg, **kwargs) for arg in args]
        elif fn_type == 'queue' or fn_type == 'remote':
            obj_ref = ray_fn.remote(*args, **kwargs)
        elif fn_type == 'repeat':
            [num_repeats, args] = args
            obj_ref = [ray_fn.remote(*args, **kwargs) for _ in range(num_repeats)]
        else:
            raise ValueError(f"fn_type {fn_type} not recognized")

        if fn_type == 'remote':
            obj_store.put_obj_ref(key=run_key, obj_ref=obj_ref)
            res = run_key
        else:
            res = ray.get(obj_ref)
    return res


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10),
                         options=[('grpc.max_send_message_length', UnaryService.MAX_MESSAGE_LENGTH),
                                  ('grpc.max_receive_message_length', UnaryService.MAX_MESSAGE_LENGTH),
                                  ])
    pb2_grpc.add_UnaryServicer_to_server(UnaryService(), server)
    server.add_insecure_port('[::]:50052')
    server.start()
    logger.info("Server up and running")
    server.wait_for_termination()


if __name__ == '__main__':
    serve()
