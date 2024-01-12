import logging
from multiprocessing import pool
from typing import List, Optional, Union

from runhouse.resources.functions import function, Function

from runhouse.resources.module import Module

logger = logging.getLogger(__name__)


class Mapper(Module):
    def __init__(
        self,
        module: Module,
        method: str,
        num_replicas: Optional[int] = -1,
        replicas: Optional[List[Module]] = None,
        **kwargs
    ):
        """
        Runhouse Mapper object. It is used for mapping a function or module method over a list of inputs,
        across a series of replicas.

        .. note::
                To create a Mapper, please use the factory method :func:`mapper`.
        """
        super().__init__(**kwargs)
        self.module = module
        self.method = method
        self.num_replicas = num_replicas
        self._auto_replicas = []
        self._user_replicas = replicas or []
        self._last_called = 0
        if self.num_replicas > len(self.replicas) and self.num_replicas > 0:
            self._add_auto_replicas(self.num_replicas - len(self.replicas))

    @property
    def replicas(self):
        return [self.module] + self._auto_replicas + self._user_replicas

    def add_replicas(self, replicas: Union[int, List[Module]]):
        if isinstance(replicas, int):
            self.num_replicas += replicas
            self._add_auto_replicas(self.num_replicas - len(self.replicas))
        else:
            self._user_replicas.extend(replicas)

    def drop_replicas(self, num_replicas: int, reap: bool = True):
        if reap:
            for replica in self._auto_replicas[-num_replicas:]:
                replica.system.kill(replica.env.name)
        self._auto_replicas = self._auto_replicas[:-num_replicas]

    def _add_auto_replicas(self, num_replicas: int):
        self._auto_replicas.extend(self.module.replicate(num_replicas))

    def increment_counter(self):
        self._last_called += 1
        if self._last_called >= len(self.replicas):
            self._last_called = 0
        return self._last_called

    @staticmethod
    def _call_method_on_replica(replica, method, args, kwargs):
        return getattr(replica, method)(*args, **kwargs)

    def map(self, *args, **kwargs):
        """Map the function or method over a list of arguments.

        Example:
            >>> def local_sum(arg1, arg2, arg3):
            >>>     return arg1 + arg2 + arg3
            >>>
            >>> remote_fn = rh.function(local_sum).to(my_cluster)
            >>> mapper = rh.mapper(remote_fn, num_replicas=2)
            >>> mapper.map([1, 2], [1, 4], [2, 3])
            >>> # output: [4, 9]

        """
        kwargs["stream_logs"] = kwargs.get("stream_logs", False)

        def call_method_on_replica(job):
            replica, method, argies, kwargies = job
            return getattr(replica, method)(*argies, **kwargies)

        jobs = [
            (self.replicas[self.increment_counter()], self.method, args, kwargs)
            for args in zip(*args)
        ]

        with pool.ThreadPool() as p:
            # Run the function in parallel on the arguments, keeping the order.
            return list(p.imap(call_method_on_replica, jobs))

    def starmap(self, args_lists: List, **kwargs):
        """Like :func:`map` except that the elements of the iterable are expected to be iterables
        that are unpacked as arguments. An iterable of ``[(1,2), (3, 4)]`` results in
        ``func(1,2), func(3,4)]``.

        Example:
            >>> def local_sum(arg1, arg2, arg3):
            >>>     return arg1 + arg2 + arg3
            >>>
            >>> remote_fn = rh.function(local_sum).to(my_cluster)
            >>> mapper = rh.mapper(remote_fn, num_replicas=2)
            >>> arg_list = [(1,2), (3, 4)]
            >>> # runs the function twice, once with args (1, 2) and once with args (3, 4)
            >>> mapper.starmap(arg_list)
        """
        kwargs["stream_logs"] = kwargs.get("stream_logs", False)

        def call_method_on_replica(job):
            replica, method, argies, kwargies = job
            return getattr(replica, method)(*argies, **kwargies)

        jobs = [
            (self.replicas[self.increment_counter()], self.method, args, kwargs)
            for args in args_lists
        ]

        with pool.ThreadPool() as p:
            # Run the function in parallel on the arguments, keeping the order.
            return list(p.imap(call_method_on_replica, jobs))

    def call(self, *args, **kwargs):
        """Call the function or method on a single replica.

        Example:
            >>> def local_sum(arg1, arg2, arg3):
            >>>     return arg1 + arg2 + arg3
            >>>
            >>> remote_fn = rh.function(local_sum).to(my_cluster)
            >>> mapper = rh.mapper(remote_fn, num_replicas=2)
            >>> for i in range(10):
            >>>     mapper.call(i, 1, 2)
            >>>     # output: 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, run in round-robin replica order

        """
        return getattr(self.replicas[self.increment_counter()], self.method)(
            *args, **kwargs
        )


def mapper(
    module: Module,
    method: Optional[str] = None,
    num_replicas: int = -1,
    replicas: Optional[List[Module]] = None,
    sync_workdir: bool = False,
    **kwargs
) -> Mapper:
    """
    A factory method for creating Mapper modules. A mapper is a module that can map a function or module method over
    a list of inputs in various ways.

    Args:
        module (Module): The module or function to be mapped.
        method (Optional[str], optional): The method of the module to be called. If the module is already a callable,
            this value defaults to ``"call"``.
        num_replicas (int, optional): The number of replicas to run the map function across. (Default: ``-1``)

            * If ``num_replicas`` is -1, it will be set to the number of available CPUs (according to Ray).
            * If ``num_replicas`` is greater than the number of user-specified replicas (``replicas``),
              the remaining replicas will be auto-generated by duplicating the module.
            * If ``num_replicas`` is 0, it will be left as the number of user-replicas passed into the module.
            * If ``num_replicas`` is less than the number of user-specified replicas, only ``num_replicas`` will
              be used.
        replicas (Optional[List[Module]], optional): List of user-specified replicas.
        sync_workdir (bool, optional): Whether to sync the working dir to the replicated environments.
            (Default: ``False``)

    Returns:
        Mapper: The resulting Mapper object.

    Example:
        >>> remote_fn = rh.function(local_fn).to(cluster)
        >>> mapper = rh.mapper(remote_fn, num_replicas=2)

        >>> remote_module = rh.module(cls=MyClass, system=cluster, env="my_env")
        >>> mapper = rh.mapper(remote_module, method=my_class_method, replicas=-1)
    """

    if callable(module) and not isinstance(module, Module):
        module = function(module, **kwargs)

    if isinstance(module, Function):
        method = method or "call"

    backup = module.env.working_dir
    if not sync_workdir:
        module.env.working_dir = None

    new_mapper = Mapper(module, method, num_replicas, replicas, **kwargs)

    if not sync_workdir:
        module.env.working_dir = backup

    return new_mapper
