import multiprocessing
import sys

from runhouse.resources.distributed.supervisor import Supervisor
from runhouse.resources.functions.function import Function

from runhouse.resources.module import Module


class RayDistributed(Supervisor):
    def __init__(self, name, module: Module = None, ray_init_options=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self._module = module
        self._ray_init_options = ray_init_options or {}

    def signature(self, rich=False):
        return self.local._module.signature(rich=rich)

    def forward(self, item, *args, **kwargs):
        from runhouse.resources.distributed.utils import subprocess_ray_fn_call_helper

        # TODO replace this with passing the filepath that this module is already writing to!
        parent_conn, child_conn = multiprocessing.Pipe()

        subproc_args = (
            self._module.fn_pointers,
            args,
            kwargs,
            child_conn,
            self._ray_init_options,
        )

        # Check if start method is already spawn, because set_start_method will error if called again
        if multiprocessing.get_start_method(allow_none=True) != "spawn":
            multiprocessing.set_start_method("spawn")

        with multiprocessing.Pool(processes=1) as pool:
            result = pool.apply_async(subprocess_ray_fn_call_helper, args=subproc_args)
            while True:
                try:
                    (msg, output_stream) = parent_conn.recv()
                    if msg == EOFError:
                        break
                    print(
                        msg,
                        end="",
                        file=sys.stdout if output_stream == "stdout" else sys.stderr,
                    )
                except EOFError:
                    break
            res = result.get()
        return res

    def __call__(self, *args, **kwargs):
        if isinstance(self._module, Function):
            return self.call(*args, **kwargs)
        else:
            raise NotImplementedError(
                "RayDistributed.__call__ can only be called on Function/Task modules."
            )
