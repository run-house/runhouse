import concurrent.futures
import logging
from typing import List, Optional, Union

try:
    from tqdm import tqdm
except ImportError:

    def tqdm(*args, **kwargs):
        return args[0]


from runhouse.resources.functions import function, Function

from runhouse.resources.module import Module

logger = logging.getLogger(__name__)


class Mapper(Module):
    def __init__(
        self,
        module: Module = None,
        method: str = None,
        replicas: Union[None, int, List[Module]] = None,
        concurrency=1,
        **kwargs,
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
        self.concurrency = concurrency
        self._auto_replicas = []
        self._user_replicas = []
        self._last_called = 0
        if isinstance(replicas, int):
            if replicas > self.num_replicas and replicas > 0:
                self.add_replicas(replicas)
        elif isinstance(replicas, list):
            self._user_replicas = replicas

    @property
    def replicas(self):
        return [self.module] + self._auto_replicas + self._user_replicas

    @property
    def num_replicas(self):
        return len(self.replicas)

    def add_replicas(self, replicas: Union[int, List[Module]]):
        if isinstance(replicas, int):
            new_replicas = replicas - self.num_replicas
            logger.info(f"Adding {new_replicas} replicas")
            self._add_auto_replicas(new_replicas)
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

    def map(self, *args, method: Optional[str] = None, retries: int = 0, **kwargs):
        """Map the function or method over a list of arguments.

        Example:
            >>> def local_sum(arg1, arg2, arg3):
            >>>     return arg1 + arg2 + arg3
            >>>
            >>> remote_fn = rh.function(local_sum).to(my_cluster)
            >>> mapper = rh.mapper(remote_fn, replicas=2)
            >>> mapper.map([1, 2], [1, 4], [2, 3])
            >>> # output: [4, 9]

        """
        # Don't stream logs by default unless the mapper is remote (i.e. mediating the mapping)
        if self.system and not self.system.on_this_cluster():
            kwargs["stream_logs"] = kwargs.get("stream_logs", True)
        else:
            kwargs["stream_logs"] = kwargs.get("stream_logs", False)

        retry_list = []

        def call_method_on_replica(job):
            replica, method_name, argies, kwargies = job
            try:
                return getattr(replica, method_name)(*argies, **kwargies)
            except Exception as e:
                logger.error(f"Error running {method_name} on {replica.name}: {e}")
                retry_list.append(job)

        jobs = [
            (
                self.replicas[self.increment_counter()],
                method or self.method,
                args,
                kwargs,
            )
            for args in zip(*args)
        ]

        results = []
        max_threads = round(self.concurrency * self.num_replicas)
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_threads) as executor:
            futs = [executor.submit(call_method_on_replica, job) for job in jobs]
            for fut in tqdm(concurrent.futures.as_completed(futs), total=len(jobs)):
                results.extend([fut.result()])
            for i in range(retries):
                if len(retry_list) == 0:
                    break
                logger.info(f"Retry {i}: {len(retry_list)} failed jobs")
                jobs, retry_list = retry_list, []
                results.extend(
                    list(
                        tqdm(
                            executor.map(call_method_on_replica, jobs), total=len(jobs)
                        )
                    )
                )

        return results

        # TODO should we add an async version of this for when we're on the cluster?
        # async def call_method_on_args(argies):
        #     return getattr(self.replicas[self.increment_counter()], self.method)(*argies, **kwargs)
        #
        # async def gather():
        #     return await asyncio.gather(
        #         *[
        #             call_method_on_args(args)
        #             for args in zip(*args)
        #         ]
        #     )
        # return asyncio.run(gather())

    def starmap(self, args_lists: List, method: Optional[str] = None, **kwargs):
        """Like :func:`map` except that the elements of the iterable are expected to be iterables
        that are unpacked as arguments. An iterable of ``[(1,2), (3, 4)]`` results in
        ``func(1,2), func(3,4)]``.

        Example:
            >>> def local_sum(arg1, arg2, arg3):
            >>>     return arg1 + arg2 + arg3
            >>>
            >>> remote_fn = rh.function(local_sum).to(my_cluster)
            >>> mapper = rh.mapper(remote_fn, replicas=2)
            >>> arg_list = [(1,2), (3, 4)]
            >>> # runs the function twice, once with args (1, 2) and once with args (3, 4)
            >>> mapper.starmap(arg_list)
        """
        # Don't stream logs by default unless the mapper is remote (i.e. mediating the mapping)
        if self.system and not self.system.on_this_cluster():
            kwargs["stream_logs"] = kwargs.get("stream_logs", True)
        else:
            kwargs["stream_logs"] = kwargs.get("stream_logs", False)

        retry_list = []

        def call_method_on_replica(job):
            replica, method_name, argies, kwargies = job
            try:
                return getattr(replica, method_name)(*argies, **kwargies)
            except Exception as e:
                logger.error(f"Error running {method_name} on {replica.name}: {e}")
                retry_list.append(job)

        jobs = [
            (
                self.replicas[self.increment_counter()],
                method or self.method,
                args,
                kwargs,
            )
            for args in args_lists
        ]

        results = []
        max_threads = round(self.concurrency * self.num_replicas)
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_threads) as executor:
            futs = [executor.submit(call_method_on_replica, job) for job in jobs]
            for fut in tqdm(concurrent.futures.as_completed(futs), total=len(jobs)):
                results.extend([fut.result()])
            for i in range(retries):
                if len(retry_list) == 0:
                    break
                logger.info(f"Retry {i}: {len(retry_list)} failed jobs")
                jobs, retry_list = retry_list, []
                results.append(
                    list(
                        tqdm(
                            executor.map(call_method_on_replica, jobs), total=len(jobs)
                        )
                    )
                )

    def call(self, *args, method: Optional[str] = None, **kwargs):
        """Call the function or method on a single replica.

        Example:
            >>> def local_sum(arg1, arg2, arg3):
            >>>     return arg1 + arg2 + arg3
            >>>
            >>> remote_fn = rh.function(local_sum).to(my_cluster)
            >>> mapper = rh.mapper(remote_fn, replicas=2)
            >>> for i in range(10):
            >>>     mapper.call(i, 1, 2)
            >>>     # output: 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, run in round-robin replica order

        """
        return getattr(self.replicas[self.increment_counter()], method or self.method)(
            *args, **kwargs
        )


def mapper(
    module: Module,
    method: Optional[str] = None,
    replicas: Union[None, int, List[Module]] = None,
    concurrency: int = 1,
    **kwargs,
) -> Mapper:
    """
    A factory method for creating Mapper modules. A mapper is a module that can map a function or module method over
    a list of inputs in various ways.

    Args:
        module (Module): The module or function to be mapped.
        method (Optional[str], optional): The method of the module to be called. If the module is already a callable,
            this value defaults to ``"call"``.
        concurrency (int, optional): The number of concurrent calls to each replica, executed in separate threads.
            Defaults to 1.
        replicas (Optional[List[Module]], optional): List of user-specified replicas, or an int specifying the number
            of replicas to be automatically created. Defaults to None.

    Returns:
        Mapper: The resulting Mapper object.

    Example:
        >>> remote_fn = rh.function(local_fn).to(cluster)
        >>> mapper = rh.mapper(remote_fn, replicas=2)

        >>> remote_module = rh.module(cls=MyClass, system=cluster, env="my_env")
        >>> mapper = rh.mapper(remote_module, method=my_class_method, replicas=-1)
    """

    if callable(module) and not isinstance(module, Module):
        module = function(module, **kwargs)

    if isinstance(module, Function):
        method = method or "call"

    return Mapper(
        module=module,
        method=method,
        replicas=replicas,
        concurrency=concurrency,
        **kwargs,
    )
