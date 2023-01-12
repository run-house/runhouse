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

import ray
import ray.cloudpickle as pickle
from sky.skylet.autostop_lib import set_last_active_time_to_now

import runhouse.grpc_handler.unary_pb2_grpc as pb2_grpc
import runhouse.grpc_handler.unary_pb2 as pb2
from runhouse.rns.packages.package import Package
from runhouse.rns.top_level_rns_fns import _set_pinned_memory_store, remove_pinned_object, flush_pinned_memory

logger = logging.getLogger(__name__)


class UnaryService(pb2_grpc.UnaryServicer):
    DEFAULT_PORT = 50052
    MAX_MESSAGE_LENGTH = 1 * 1024 * 1024 * 1024  # 1 GB

    def __init__(self, *args, **kwargs):
        ray.init(address="auto")

        self.sky_data = {}
        with open(os.path.expanduser("~/.sky/sky_ray.yml")) as f:
            self.sky_data = yaml.safe_load(f)
        self.cluster_name = self.sky_data.get('cluster_name', 'cluster')

        # Name the file cluster_name-<YYYY-MM-DD_HH-MM-SS-PID.txt>
        # Need to do this inside init or log files are created on the user's local machine
        log_filename = f"{self.cluster_name}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_p{os.getpid()}.txt"
        log_path = Path("~/.rh/logs").expanduser()
        log_path.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(log_path / log_filename)
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        self.imported_modules = {}
        self._installed_packages = []
        self._shared_objects = {}
        _set_pinned_memory_store(self._shared_objects)
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

    def ClearPins(self, request, context):
        pins_to_clear = pickle.loads(request.message)
        logger.info(f"Message received from client to clear pins: {pins_to_clear or 'all'}")
        cleared = []
        if pins_to_clear:
            for pin in pins_to_clear:
                remove_pinned_object(pin)
                cleared.append(pin)
        else:
            cleared = list(self._shared_objects.keys())
            flush_pinned_memory()

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
                    self.imported_modules[module_name] = importlib.reload(self.imported_modules[module_name])
                    logger.info(f"Reloaded module {module_name}")
                else:
                    self.imported_modules[module_name] = importlib.import_module(module_name)
                    logger.info(f"Importing module {module_name}")
                fn = getattr(self.imported_modules[module_name], fn_name)

            # TODO other possible fn_types: 'batch', 'streaming'
            if fn_type == 'call':
                args = [ray.get(arg) if isinstance(arg, ray.ObjectRef) else arg for arg in args]
                res = fn(*args, **kwargs)
            elif fn_type == 'get':
                obj_ref = args[0]
                res = ray.get(obj_ref)
            else:
                ray_fn = ray.remote(num_cpus=0, num_gpus=0,
                                    runtime_env={"env_vars": {"PYTHONPATH": module_path or ''}})(fn)
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
                    res = obj_ref
                else:
                    res = ray.get(obj_ref)
            # [res, None, None] is a silly hack for packaging result alongside exception and traceback
            result = {'message': pickle.dumps([res, None, None]), 'received': True}

            # TODO [DG]: If response contains data resources with local data, convert the provider and address
            # to SFTP:ip so the data can be accessed on the destination.

            self.register_activity()
            return pb2.MessageResponse(**result)
        except Exception as e:
            logger.exception(e)
            message = [None, e, traceback.format_exc()]
            self.register_activity()
            return pb2.MessageResponse(message=pickle.dumps(message), received=False)


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
