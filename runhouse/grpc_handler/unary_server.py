import importlib
import sys
import logging
from pathlib import Path
import grpc
from concurrent import futures
import traceback

import ray
import ray.cloudpickle as pickle

import runhouse.grpc_handler.unary_pb2_grpc as pb2_grpc
import runhouse.grpc_handler.unary_pb2 as pb2
from runhouse.rns.package import Package
from runhouse.rns.top_level_rns_fns import _set_pinned_memory_store

logger = logging.getLogger(__name__)

class UnaryService(pb2_grpc.UnaryServicer):
    DEFAULT_PORT = 50052

    def __init__(self, *args, **kwargs):
        ray.init(address="auto")
        self.fn_module = None
        self._installed_packages = []
        self._shared_objects = {}
        _set_pinned_memory_store(self._shared_objects)

    def InstallPackages(self, request, context):
        logger.info(f"Message received from client: {request.message}")
        packages = pickle.loads(request.message)
        for package in packages:
            if isinstance(package, str):
                pkg = Package.from_string(package)
            elif hasattr(package, 'install'):
                pkg = package
            else:
                raise ValueError(f"package {package} not recognized")

            if pkg.name in self._installed_packages:
                continue
            logger.info(f"Installing package: {pkg.name}")
            pkg.install()
            self._installed_packages.append(pkg.name)
        return pb2.MessageResponse(message=pickle.dumps(True), received=True)

    def GetServerResponse(self, request, context):

        # get the function result from the incoming request
        logger.info(f"Message received from client: {request.message}")
        try:
            [relative_path, module_name, fn_name, fn_type, args, kwargs] = pickle.loads(request.message)

            module_path = None
            # If relative_path is None, the module is not in the working dir, and should be in the reqs
            if relative_path:
                module_path = str((Path.home() / relative_path).resolve())
                sys.path.append(module_path)

            if module_name == "notebook":
                fn = fn_name  # Already unpickled above
            else:
                if self.fn_module and self.fn_module.__name__ == module_name:
                    self.fn_module = importlib.reload(self.fn_module)
                else:
                    self.fn_module = importlib.import_module(module_name)
                fn = getattr(self.fn_module, fn_name)

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

            return pb2.MessageResponse(**result)
        except Exception as e:
            logger.exception(e)
            message = [None, e, traceback.format_exc()]
            return pb2.MessageResponse(message=pickle.dumps(message), received=False)


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    pb2_grpc.add_UnaryServicer_to_server(UnaryService(), server)
    server.add_insecure_port('[::]:50052')
    server.start()
    logger.info("Server up and running")
    server.wait_for_termination()


if __name__ == '__main__':
    serve()
