import inspect
import logging
from pathlib import Path
from typing import Any, List, Optional, Tuple, Union

from runhouse import globals
from runhouse.resources.envs import Env
from runhouse.resources.hardware import Cluster
from runhouse.resources.module import Module

from runhouse.resources.resource import Resource

logger = logging.getLogger(__name__)


class Function(Module):
    RESOURCE_TYPE = "function"

    def __init__(
        self,
        fn_pointers: Optional[Tuple] = None,
        name: Optional[str] = None,
        system: Optional[Cluster] = None,
        env: Optional[Env] = None,
        dryrun: bool = False,
        **kwargs,  # We have this here to ignore extra arguments when calling from from_config
    ):
        """
        Runhouse Function object. It is comprised of the entrypoint, system/cluster,
        and dependencies necessary to run the service.

        .. note::
                To create a Function, please use the factory method :func:`function`.
        """
        self.fn_pointers = fn_pointers
        self._loaded_fn = None
        super().__init__(name=name, dryrun=dryrun, system=system, env=env, **kwargs)

    # ----------------- Constructor helper methods -----------------

    @classmethod
    def from_config(cls, config: dict, dryrun: bool = False, _resolve_children=True):
        """Create a Function object from a config dictionary."""
        if isinstance(config["system"], dict):
            config["system"] = Cluster.from_config(
                config["system"], dryrun=dryrun, _resolve_children=_resolve_children
            )
        if isinstance(config["env"], dict):
            config["env"] = Env.from_config(
                config["env"], dryrun=dryrun, _resolve_children=_resolve_children
            )

        config.pop("resource_subtype", None)
        return Function(**config, dryrun=dryrun)

    def share(self, *args, visibility=None, **kwargs):
        if visibility and not visibility == self.visibility:
            self.visibility = visibility
            super().remote.visibility = (
                visibility  # do this to avoid hitting Function's .remote
            )
        return super().share(*args, **kwargs, visibility=visibility)

    def default_name(self):
        return (
            self.fn_pointers[2] if self.fn_pointers else None
        ) or super().default_name()

    def to(
        self,
        system: Union[str, Cluster],
        env: Optional[Union[str, List[str], Env]] = None,
        name: Optional[str] = None,
        force_install: bool = False,
    ):
        """to(system: str | Cluster | None = None, env: List[str] | Env = [], force_install: bool = False)

        Set up a Function and Env on the given system.
        If the function is sent to AWS, the system should be ``aws_lambda``
        See the args of the factory method :func:`function` for more information.

        Example:
            >>> rh.function(fn=local_fn).to(gpu_cluster)
            >>> rh.function(fn=local_fn).to(system=gpu_cluster, env=my_conda_env)
            >>> rh.function(fn=local_fn).to(system='aws_lambda')  # will deploy the rh.function to AWS as a Lambda.
        """  # noqa: E501

        if isinstance(system, str) and system.lower() == "lambda_function":
            from runhouse.resources.functions.aws_lambda_factory import aws_lambda_fn

            return aws_lambda_fn(
                fn=self._get_obj_from_pointers(*self.fn_pointers), env=env
            )

        return super().to(
            system=system, env=env, name=name, force_install=force_install
        )

    # ----------------- Function call methods -----------------

    def __call__(self, *args, **kwargs) -> Any:
        """Call the function on its system

        Args:
             *args: Optional args for the Function
             stream_logs (bool): Whether to stream the logs from the Function's execution.
                Defaults to ``True``.
             run_name (Optional[str]): Name of the Run to create. If provided, a Run will be created
                for this function call, which will be executed synchronously on the cluster before returning its result
             **kwargs: Optional kwargs for the Function

        Returns:
            The Function's return value
        """
        return self.call(*args, **kwargs)

    def call(self, *args, **kwargs) -> Any:
        # We need this strictly because Module's __getattribute__ overload can't pick up the __call__ method
        if not self._loaded_fn:
            self._loaded_fn = self._get_obj_from_pointers(*self.fn_pointers)
        return self._loaded_fn(*args, **kwargs)

    def method_signature(self, method):
        if callable(method) and method.__name__ == "call":
            return self.method_signature(self._get_obj_from_pointers(*self.fn_pointers))
        return super().method_signature(method)

    def map(self, *args, **kwargs):
        """Map a function over a list of arguments.

        Example:
            >>> def local_sum(arg1, arg2, arg3):
            >>>     return arg1 + arg2 + arg3
            >>>
            >>> remote_fn = rh.function(local_fn).to(gpu)
            >>> remote_fn.map([1, 2], [1, 4], [2, 3])
            >>> # output: [4, 9]

        """
        import ray

        fn = self._get_obj_from_pointers(*self.fn_pointers)
        ray_wrapped_fn = ray.remote(fn)
        return ray.get([ray_wrapped_fn.remote(*args, **kwargs) for args in zip(*args)])

    def starmap(self, args_lists, **kwargs):
        """Like :func:`map` except that the elements of the iterable are expected to be iterables
        that are unpacked as arguments. An iterable of [(1,2), (3, 4)] results in [func(1,2), func(3,4)].

        Example:
            >>> arg_list = [(1,2), (3, 4)]
            >>> # runs the function twice, once with args (1, 2) and once with args (3, 4)
            >>> remote_fn.starmap(arg_list)
        """
        import ray

        fn = self._get_obj_from_pointers(*self.fn_pointers)
        ray_wrapped_fn = ray.remote(fn)
        return ray.get([ray_wrapped_fn.remote(*args, **kwargs) for args in args_lists])

    def run(self, *args, local=True, **kwargs):
        key = self.call.run(*args, **kwargs)
        return key

    def get(self, run_key):
        """Get the result of a Function call that was submitted as async using `run`.

        Args:
            run_key: A single or list of runhouse run_key strings returned by calling ``.call.remote()`` on the
                Function. The ObjectRefs must be from the cluster that this Function is running on.

        Example:
            >>> remote_fn = rh.function(local_fn).to(gpu)
            >>> remote_fn_run = remote_fn.run()
            >>> remote_fn.get(remote_fn_run.name)
        """
        return self.system.get(run_key)

    def config(self, condensed=True):
        config = super().config(condensed)
        config.update(
            {
                "fn_pointers": self.fn_pointers,
            }
        )
        return config

    def send_secrets(self, providers: Optional[List[str]] = None):
        """Send secrets to the system.

        Example:
            >>> remote_fn.send_secrets(providers=["aws", "lambda"])
        """
        self.system.sync_secrets(providers=providers)

    def http_url(self, curl_command=False, *args, **kwargs) -> str:
        """
        Return the endpoint needed to run the Function on the remote cluster, or provide the curl command if requested.
        """
        raise NotImplementedError("http_url not yet implemented for Function")

    def notebook(self, persist=False, sync_package_on_close=None, port_forward=8888):
        """Tunnel into and launch notebook from the system."""
        if self.system is None:
            raise RuntimeError("Cannot SSH, running locally")

        self.system.notebook(
            persist=persist,
            sync_package_on_close=sync_package_on_close,
            port_forward=port_forward,
        )

    def get_or_call(self, run_name: str, load=True, local=True, *args, **kwargs) -> Any:
        """Check if object already exists on cluster or rns, and if so return the result. If not, run the function.
        Keep in mind this can be called with any of the usual method call modifiers - `remote=True`, `run_async=True`,
        `stream_logs=False`, etc.

        Args:
            run_name (Optional[str]): Name of a particular run for this function.
                If not provided will use the function's name.
            load (bool): Whether to load the name from the RNS if it exists.
            *args: Arguments to pass to the function for the run (relevant if creating a new run).
            **kwargs: Keyword arguments to pass to the function for the run (relevant if creating a new run).

        Returns:
            Any: Result of the Run

        Example:
            >>> # previously, remote_fn.run(arg1, arg2, run_name="my_async_run")
            >>> remote_fn.get_or_call()
        """
        # TODO let's just do this for functions initially, and decide if we want to support it for calls on modules
        #  as well. Right now this only works with remote=True, we should decide if we want to fix that later.

        if load:
            resource = globals.rns_client.load_config(name=run_name)
            if resource:
                return Resource.from_name(name=run_name, dryrun=self.dryrun)
        try:
            return self.system.get(run_name, default=KeyError)
        except KeyError:
            logger.info(f"Item {run_name} not found on cluster. Running function.")

        return self.call(*args, **kwargs, run_name=run_name)

    def keep_warm(
        self,
        autostop_mins=None,
    ):
        """Keep the system warm for autostop_mins. If autostop_mins is ``None`` or -1, keep warm indefinitely.

        Example:
            >>> # keep gpu warm for 30 mins
            >>> remote_fn = rh.function(local_fn).to(gpu)
            >>> remote_fn.keep_warm(autostop_mins=30)
        """
        if autostop_mins is None:
            logger.info(f"Keeping {self.name} indefinitely warm")
            # keep indefinitely warm if user doesn't specify
            autostop_mins = -1
        self.system.keep_warm(autostop_mins=autostop_mins)

        return self

    @staticmethod
    def _handle_nb_fn(fn, fn_pointers, serialize_notebook_fn, name):
        """Handle the case where the user passes in a notebook function"""
        if serialize_notebook_fn:
            # This will all be cloudpickled by the RPC client and unpickled by the RPC server
            # Note that this means the function cannot be saved, and it's better that way because
            # pickling functions is not meant for long term storage. Case in point, this method will be
            # sensitive to differences in minor Python versions between the serializing and deserializing envs.
            return "", "notebook", fn
        else:
            module_path = Path.cwd() / (f"{name}_fn.py" if name else "sent_fn.py")
            logging.info(
                f"Writing out function to {str(module_path)}. Please make "
                f"sure the function does not rely on any local variables, "
                f"including imports (which should be moved inside the function body)."
            )
            if not name:
                logging.warning(
                    "You should name Functions that are created in notebooks to avoid naming collisions "
                    "between the modules that are created to hold their functions "
                    '(i.e. "sent_fn.py" errors.'
                )
            source = inspect.getsource(fn).strip()
            with module_path.open("w") as f:
                f.write(source)
            return fn_pointers[0], module_path.stem, fn_pointers[2]
