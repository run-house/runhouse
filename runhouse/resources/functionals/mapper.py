import concurrent.futures
import contextvars
import logging
from typing import Callable, List, Optional, Union

try:
    from tqdm import tqdm
except ImportError:

    def tqdm(*args, **kwargs):
        return args[0]


from runhouse.resources.envs.env import Env
from runhouse.resources.functions import function, Function
from runhouse.resources.hardware.cluster import Cluster

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
        self._num_auto_replicas = None
        self._auto_replicas = []
        self._user_replicas = []
        self._last_called = 0
        if isinstance(replicas, int):
            if self.module.system:
                # Only add replicas if the replicated module is already on a cluster
                if replicas > self.num_replicas and replicas > 0:
                    self.add_replicas(replicas)
            else:
                # Otherwise, store this for later once we've sent the mapper to the cluster
                self._num_auto_replicas = replicas
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

    def to(
        self,
        system: Union[str, Cluster],
        env: Optional[Union[str, List[str], Env]] = None,
        name: Optional[str] = None,
        force_install: bool = False,
    ):
        """Put a copy of the Mapper and its internal module on the destination system and env, and
        return the new mapper.

        Example:
            >>> local_mapper = rh.mapper(my_module, replicas=2)
            >>> cluster_mapper = local_mapper.to(my_cluster)
        """
        if not self.module.system:
            # Note that we don't pass name here, as this is the name meant for the mapper
            self.module = self.module.to(
                system=system, env=env, force_install=force_install
            )
        remote_mapper = super().to(
            system=system, env=env, name=name, force_install=force_install
        )

        if isinstance(self._num_auto_replicas, int):
            remote_mapper.add_replicas(self._num_auto_replicas)
        return remote_mapper

    def map(self, *args, method: Optional[str] = None, retries: int = 0, **kwargs):
        """Map the function or method over a list of arguments.

        Example:
            >>> mapper = rh.mapper(local_sum, replicas=2).to(my_cluster)
            >>> mapper.map([1, 2], [1, 4], [2, 3], retries=3)

            >>> # If you're mapping over a remote module, you can choose not to specify which method to call initially
            >>> # so you can call different methods in different maps (note that our replicas can hold state!)
            >>> # Note that in the example below we're careful to use the same number of replicas as data
            >>> # we have to process, or the state in a replica would be overwritten by the next call.
            >>> shards = len(source_paths)
            >>> mapper = rh.mapper(remote_module, replicas=shards).to(my_cluster)
            >>> mapper.map(*source_paths, method="load_data")
            >>> mapper.map([]*shards, method="process_data")  # Calls each replica once with no args
            >>> mapper.map(*output_paths, method="save_data")
        """
        # Don't stream logs by default unless the mapper is remote (i.e. mediating the mapping)
        return self.starmap(
            arg_list=zip(*args), method=method, retries=retries, **kwargs
        )

    def starmap(
        self, arg_list: List, method: Optional[str] = None, retries: int = 0, **kwargs
    ):
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

        def call_method_on_replica(job, retry=True):
            replica, method_name, context, argies, kwargies = job
            # reset context
            for var, value in context.items():
                var.set(value)

            try:
                return getattr(replica, method_name)(*argies, **kwargies)
            except Exception as e:
                logger.error(f"Error running {method_name} on {replica.name}: {e}")
                if retry:
                    retry_list.append(job)
                else:
                    return e

        context = contextvars.copy_context()
        jobs = [
            (
                self.replicas[self.increment_counter()],
                method or self.method,
                context,
                args,
                kwargs,
            )
            for args in arg_list
        ]

        results = []
        max_threads = round(self.concurrency * self.num_replicas)
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_threads) as executor:
            futs = [
                executor.submit(call_method_on_replica, job, retries > 0)
                for job in jobs
            ]
            for fut in tqdm(concurrent.futures.as_completed(futs), total=len(jobs)):
                results.extend([fut.result()])
            for i in range(retries):
                if len(retry_list) == 0:
                    break
                logger.info(f"Retry {i}: {len(retry_list)} failed jobs")
                jobs, retry_list = retry_list, []
                retry = i != retries - 1
                results.append(
                    list(
                        tqdm(
                            executor.map(call_method_on_replica, jobs, retry),
                            total=len(jobs),
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
    module: Union[Module, Callable],
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
        >>> def local_sum(arg1, arg2, arg3):
        >>>     return arg1 + arg2 + arg3
        >>>
        >>> # Option 1: Pass a function directly to the mapper, and send both to the cluster
        >>> mapper = rh.mapper(local_sum, replicas=2).to(my_cluster)
        >>> mapper.map([1, 2], [1, 4], [2, 3])

        >>> # Option 2: Create a remote module yourself and pass it to the mapper, which is still local
        >>> remote_fn = rh.function(local_sum).to(my_cluster, env=my_fn_env)
        >>> mapper = rh.mapper(remote_fn, replicas=2)
        >>> mapper.map([1, 2], [1, 4], [2, 3])
        >>> # output: [4, 9]

        >>> # Option 3: Create a remote module and mapper for greater flexibility, and send both to the cluster
        >>> # You can map over a "class" module (stateless) or an "instance" module to preserve state
        >>> remote_class = rh.module(cls=MyClass).to(system=cluster, env=my_module_env)
        >>> stateless_mapper = rh.mapper(remote_class, method="my_class_method", replicas=2).to(cluster)
        >>> mapper.map([1, 2], [1, 4], [2, 3])

        >>> remote_app = remote_class()
        >>> stateful_mapper = rh.mapper(remote_app, method="my_instance_method", replicas=2).to(cluster)
        >>> mapper.map([1, 2], [1, 4], [2, 3])
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
