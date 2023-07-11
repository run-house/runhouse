import copy
import inspect
import json
import logging
import os
import re
import time
import warnings
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import requests

from runhouse import rh_config
from runhouse.rns.api_utils.resource_access import ResourceAccess
from runhouse.rns.api_utils.utils import load_resp_content, read_resp_data
from runhouse.rns.envs import CondaEnv, Env
from runhouse.rns.hardware import Cluster
from runhouse.rns.packages import git_package, Package

from runhouse.rns.resource import Resource
from runhouse.rns.run_module_utils import call_fn_by_type

from runhouse.rns.utils.env import _env_vars_from_file, _get_env_from
from runhouse.rns.utils.hardware import _get_cluster_from

logger = logging.getLogger(__name__)


class Function(Resource):
    RESOURCE_TYPE = "function"
    DEFAULT_ACCESS = "write"

    def __init__(
        self,
        fn_pointers: Optional[Tuple] = None,
        system: Optional[Cluster] = None,
        name: Optional[str] = None,
        env: Optional[Env] = None,
        dryrun: bool = False,
        access: Optional[str] = None,
        resources: Optional[dict] = None,
        **kwargs,  # We have this here to ignore extra arguments when calling from from_config
    ):
        """
        Runhouse Function object. It comprises of the entrypoint, system/cluster,
        and dependencies necessary to run the service.

        .. note::
                To create a Function, please use the factory method :func:`function`.
        """
        self.fn_pointers = fn_pointers
        self.system = system
        self.env = env
        self.access = access or self.DEFAULT_ACCESS
        self.dryrun = dryrun
        self.resources = resources or {}
        super().__init__(name=name, dryrun=dryrun)

        if not self.dryrun:
            self = self.to(self.system, env=self.env)

    # ----------------- Constructor helper methods -----------------

    @staticmethod
    def from_config(config: dict, dryrun: bool = False):
        """Create a Function object from a config dictionary."""
        if isinstance(config["system"], dict):
            config["system"] = Cluster.from_config(config["system"], dryrun=dryrun)
        if isinstance(config["env"], dict):
            config["env"] = Env.from_config(config["env"], dryrun=dryrun)

        return Function(**config, dryrun=dryrun)

    @classmethod
    def _check_for_child_configs(cls, config):
        """Overload by child resources to load any resources they hold internally."""
        system = config["system"]
        if isinstance(system, str):
            config["system"] = rh_config.rns_client.load_config(name=system)
            # if the system is set to a cluster
            if not config["system"]:
                raise Exception(f"No cluster config saved for {system}")

        config["env"] = _get_env_from(config["env"])
        return config

    def to(
        self,
        system: Union[str, Cluster] = None,
        env: Union[List[str], Env] = [],
        # Variables below are deprecated
        reqs: Optional[List[str]] = None,
        setup_cmds: Optional[List[str]] = [],
    ):
        """
        Set up a Function and Env on the given system.

        See the args of the factory method :func:`function` for more information.

        Example:
            >>> rh.function(fn=local_fn).to(gpu_cluster)
            >>> rh.function(fn=local_fn).to(system=gpu_cluster, env=my_conda_env)
        """
        if setup_cmds:
            warnings.warn(
                "``setup_cmds`` argument has been deprecated. "
                "Please pass in setup commands to the ``Env`` class corresponding to the function instead."
            )

        # to retain backwards compatibility
        if reqs or setup_cmds:
            warnings.warn(
                "``reqs`` and ``setup_cmds`` arguments has been deprecated. Please use ``env`` instead."
            )
            env = Env(reqs=reqs, setup_cmds=setup_cmds)
        elif env and isinstance(env, List):
            env = Env(reqs=env, setup_cmds=setup_cmds)
        else:
            env = env or self.env or Env()
            env = _get_env_from(env)

        if (
            self.dryrun
            or not (system or self.system)
            or self.access not in ["write", "read"]
        ):
            # don't move the function to a system
            self.env = env
            return self

        # We need to backup the system here so the __getstate__ method of the cluster
        # doesn't wipe the client of this function's cluster when deepcopy copies it.
        hw_backup = self.system
        self.system = None
        new_function = copy.deepcopy(self)
        self.system = hw_backup

        if system:
            if isinstance(system, str):
                system = Cluster.from_name(system, dryrun=self.dryrun)
            new_function.system = system
        else:
            new_function.system = self.system

        logging.info("Setting up Function on cluster.")
        # To up cluster in case it's not yet up
        new_function.system.check_server()
        new_env = env.to(new_function.system)
        logging.info("Function setup complete.")
        new_function.env = new_env

        return new_function

    @staticmethod
    def _extract_fn_paths(raw_fn: Callable, reqs: List[str]):
        """Get the path to the module, module name, and function name to be able to import it on the server"""
        if not isinstance(raw_fn, Callable):
            raise TypeError(
                f"Invalid fn for Function, expected Callable but received {type(raw_fn)}"
            )
        # Background on all these dunders: https://docs.python.org/3/reference/import.html
        module = inspect.getmodule(raw_fn)

        # Need to resolve in case just filename is given
        module_path = (
            str(Path(inspect.getfile(module)).resolve())
            if hasattr(module, "__file__")
            else None
        )

        # TODO better way of detecting if in a notebook or interactive Python env
        if (
            not module_path
            or module_path.endswith("ipynb")
            or raw_fn.__name__ == "<lambda>"
        ):
            # The only time __file__ wouldn't be present is if the function is defined in an interactive
            # interpreter or a notebook. We can't import on the server in that case, so we need to cloudpickle
            # the fn to send it over. The __call__ function will serialize the function if we return it this way.
            # This is a short-term hack.
            # return None, "notebook", raw_fn.__name__
            root_path = os.getcwd()
            module_name = "notebook"
            fn_name = raw_fn.__name__
        else:
            root_path = os.path.dirname(module_path)
            module_name = inspect.getmodulename(module_path)
            # TODO __qualname__ doesn't work when fn is aliased funnily, like torch.sum
            fn_name = getattr(raw_fn, "__qualname__", raw_fn.__name__)

            # Adapted from https://github.com/modal-labs/modal-client/blob/main/modal/_function_utils.py#L94
            if getattr(module, "__package__", None):
                module_path = os.path.abspath(module.__file__)
                package_paths = [
                    os.path.abspath(p) for p in __import__(module.__package__).__path__
                ]
                base_dirs = [
                    base_dir
                    for base_dir in package_paths
                    if os.path.commonpath((base_dir, module_path)) == base_dir
                ]

                if len(base_dirs) != 1:
                    logger.info(f"Module files: {module_path}")
                    logger.info(f"Package paths: {package_paths}")
                    logger.info(f"Base dirs: {base_dirs}")
                    raise Exception("Wasn't able to find the package directory!")
                root_path = os.path.dirname(base_dirs[0])
                module_name = module.__spec__.name

        remote_import_path = None
        for req in reqs:
            local_path = None
            if (
                isinstance(req, Package)
                and not isinstance(req.install_target, str)
                and req.install_target.is_local()
            ):
                local_path = Path(req.install_target.local_path)
            elif isinstance(req, str):
                if req.split(":")[0] in ["local", "reqs", "pip"]:
                    req = req.split(":")[1]

                if Path(req).expanduser().resolve().exists():
                    # Relative paths are relative to the working directory in Folders/Packages!
                    local_path = (
                        Path(req).expanduser()
                        if Path(req).expanduser().is_absolute()
                        else Path(rh_config.rns_client.locate_working_dir()) / req
                    )

            if local_path:
                try:
                    # Module path relative to package
                    remote_import_path = str(
                        local_path.name / Path(root_path).relative_to(local_path)
                    )
                    break
                except ValueError:  # Not a subdirectory
                    pass

        return remote_import_path, module_name, fn_name

    # Found in python decorator logic, maybe use
    # func_name = getattr(f, '__qualname__', f.__name__)
    # module_name = getattr(f, '__module__', '')
    # if module_name:
    #     full_name = f'{module_name}.{func_name}'
    # else:
    #     full_name = func_name

    # ----------------- Function call methods -----------------

    def __call__(self, *args, stream_logs=True, run_name: str = None, **kwargs) -> Any:
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

        fn_type = "call"
        if self.access in [ResourceAccess.WRITE, ResourceAccess.READ]:
            run_locally = not self.system or self.system.on_this_cluster()
            if not run_locally:
                start = time.time()
                run_obj = self.run(*args, run_name=run_name, **kwargs)
                res = self.system.get(run_obj.name, stream_logs=stream_logs)
                end = time.time()
                logging.info(
                    f"Time to call remote function: {round(end - start, 2)} seconds"
                )
                return res
            else:
                [relative_path, module_name, fn_name] = self.fn_pointers
                conda_env = (
                    self.env.env_name
                    if self.env and isinstance(self.env, CondaEnv)
                    else None
                )
                env_vars = self.env.env_vars
                # If we're on this cluster, don't pickle the result before passing back.
                # We need to pickle before passing back in most cases because the env in
                # which the function executes may have a different set of packages than the
                # server, so when Ray passes a result back into the server it will may fail to
                # unpickle. We assume the user's client has the necessary packages to unpickle
                # their own result.

                return call_fn_by_type(
                    fn_type=fn_type,
                    fn_name=fn_name,
                    relative_path=relative_path,
                    module_name=module_name,
                    resources=self.resources,
                    conda_env=conda_env,
                    env_vars=env_vars,
                    run_name=run_name,
                    args=args,
                    kwargs=kwargs,
                    serialize_res=False,
                )
        else:
            # run the function via http path - user only needs Proxy access
            if self.access != ResourceAccess.PROXY:
                raise RuntimeError("Running http path requires proxy access")
            if not rh_config.rns_client.token:
                raise ValueError(
                    "Token must be saved in the local .rh config in order to use an http path"
                )
            http_url = self.http_url()
            logger.info(f"Running {self.name} via http path: {http_url}")
            resp = requests.post(
                http_url,
                data=json.dumps({"args": args, "kwargs": kwargs}),
                headers=rh_config.rns_client.request_headers,
            )
            if resp.status_code != 200:
                raise Exception(
                    f"Failed to run Function endpoint: {load_resp_content(resp)}"
                )

            return read_resp_data(resp)

    def repeat(self, num_repeats: int, *args, **kwargs):
        """Repeat the Function call multiple times.

        Args:
            num_repeats (int): Number of times to repeat the Function call.
            *args: Positional arguments to pass to the Function
            **kwargs: Keyword arguments to pass to the Function

        Example:
            >>> remote_fn = rh.function(local_fn).to(gpu)
            >>> remote_fn.repeat(num_repeats=5)
        """
        if self.access in [ResourceAccess.WRITE, ResourceAccess.READ]:
            return self._call_fn_with_ssh_access(
                fn_type="repeat", args=[num_repeats, args], kwargs=kwargs
            )
        else:
            raise NotImplementedError(
                "Function.repeat only works with Write or Read access, not Proxy access"
            )

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
        arg_list = zip(*args)
        if self.access in [ResourceAccess.WRITE, ResourceAccess.READ]:
            return self._call_fn_with_ssh_access(
                fn_type="starmap", args=arg_list, kwargs=kwargs
            )
        else:
            raise NotImplementedError(
                "Function.map only works with Write or Read access, not Proxy access"
            )

    def starmap(self, args_lists, **kwargs):
        """Like :func:`map` except that the elements of the iterable are expected to be iterables
        that are unpacked as arguments. An iterable of [(1,2), (3, 4)] results in [func(1,2), func(3,4)].

        Example:
            >>> arg_list = [(1,2), (3, 4)]
            >>> # runs the function twice, once with args (1, 2) and once with args (3, 4)
            >>> remote_fn.starmap(arg_list)
        """
        if self.access in [ResourceAccess.WRITE, ResourceAccess.READ]:
            return self._call_fn_with_ssh_access(
                fn_type="starmap", args=args_lists, kwargs=kwargs
            )
        else:
            raise NotImplementedError(
                "Function.starmap only works with Write or Read access, not Proxy access"
            )

    def enqueue(self, resources: Optional[Dict] = None, *args, **kwargs):
        """
        Enqueue a Function call to be run later. This ensures a function call doesn’t run simultaneously with other
        calls, but will wait until the execution completes.

        Example:
            >>> # This will run the functions sequentially
            >>> [remote_fn.enqueue() for _ in range(3)]
        """
        # Add resources one-off without setting as a Function param
        if self.access in [ResourceAccess.WRITE, ResourceAccess.READ]:
            return self._call_fn_with_ssh_access(
                fn_type="queue",
                resources=resources or self.resources,
                args=args,
                kwargs=kwargs,
            )
        else:
            raise NotImplementedError(
                "Function.enqueue only works with Write or Read access, not Proxy access"
            )

    def remote(self, *args, **kwargs):
        warnings.warn("`remote()` is deprecated, use `run()` instead")
        run_obj = self.run(*args, **kwargs)
        return run_obj.name

    def run(self, *args, run_name: str = None, **kwargs):
        """Run async remote call on cluster.

        Args:
            *args: Optional args for the Function
            run_name (Optional[str]): Name of the Run to create. If not provided, a name will automatically
             be generated.
             **kwargs: Optional kwargs for the Function
        Returns:
            Run: Run object
        Example:
            >>> remote_fn = rh.function(local_fn).to(gpu)
            >>> remote_fn.run(arg1, arg2, run_name="my_async_run")
        """
        if self.access in [ResourceAccess.WRITE, ResourceAccess.READ]:
            from runhouse import Run

            # Use the run_name if provided, otherwise create a new one using the Function's name
            run_name = run_name or Run._create_new_run_name(self.name)

            run_obj = self._call_fn_with_ssh_access(
                fn_type="remote", run_name=run_name, args=args, kwargs=kwargs
            )

            logger.info(f"Submitted remote call to cluster for {run_name}")
            return run_obj
        else:
            raise NotImplementedError(
                "Function.remote only works with Write or Read access, not Proxy access"
            )

    def get(self, run_key):
        """Get the result of a Function call that was submitted as async using `remote`.

        Args:
            run_key: A single or list of runhouse run_key strings returned by a Function.remote() call. The ObjectRefs
                must be from the cluster that this Function is running on.

        Example:
            >>> remote_fn = rh.function(local_fn).to(gpu)
            >>> remote_fn_run = remote_fn.run()
            >>> remote_fn.get(remote_fn_run.name)
        """
        return self.system.get(run_key)

    def _call_fn_with_ssh_access(
        self, fn_type, resources=None, run_name=None, args=None, kwargs=None
    ):
        # https://docs.ray.io/en/latest/ray-core/tasks/patterns/map-reduce.html
        # return ray.get([map.remote(i, map_func) for i in replicas])
        # TODO allow specifying resources per worker for map
        # TODO [DG] check whether we're on the cluster and if so, just call the function directly via the
        # helper function currently in UnaryServer
        resources = (
            resources or self.resources
        )  # Allow for passing in one-off resources for this specific call
        name = self.name or "anonymous function"
        if self.fn_pointers is None:
            raise RuntimeError(f"No fn pointers saved for {name}")

        [relative_path, module_name, fn_name] = self.fn_pointers

        name = self.name or fn_name or "anonymous function"
        logger.info(f"Running {name} via HTTP")
        env_name = (
            self.env.env_name if (self.env and isinstance(self.env, CondaEnv)) else None
        )
        env_vars = self.env.env_vars if self.env else {}
        if not isinstance(env_vars, dict):
            env_vars = _env_vars_from_file(env_vars)

        res = self.system._run_module(
            relative_path,
            module_name,
            fn_name,
            fn_type,
            resources,
            env_name,
            env_vars,
            run_name,
            args,
            kwargs,
        )
        return res

    # TODO [DG] test this properly
    # def debug(self, redirect_logging=False, timeout=10000, *args, **kwargs):
    #     """Run the Function in debug mode. This will run the Function through a tunnel interpreter, which
    #     allows the use of breakpoints and other debugging tools, like rh.ipython().
    #     FYI, alternative ideas from Ray: https://github.com/ray-project/ray/issues/17197
    #     FYI, alternative Modal folks shared: https://github.com/modal-labs/modal-client/pull/32
    #     """
    #     from paramiko import AutoAddPolicy
    #
    #     # Importing this here because they're heavy
    #     from plumbum.machines.paramiko_machine import ParamikoMachine
    #     from rpyc.utils.classic import redirected_stdio
    #     from rpyc.utils.zerodeploy import DeployedServer
    #
    #     creds = self.system.ssh_creds()
    #     ssh_client = ParamikoMachine(
    #         self.system.address,
    #         user=creds["ssh_user"],
    #         keyfile=str(Path(creds["ssh_private_key"]).expanduser()),
    #         missing_host_policy=AutoAddPolicy(),
    #     )
    #     server = DeployedServer(
    #         ssh_client, server_class="rpyc.utils.server.ForkingServer"
    #     )
    #     conn = server.classic_connect()
    #
    #     if redirect_logging:
    #         rlogger = conn.modules.logging.getLogger()
    #         rlogger.parent = logging.getLogger()
    #
    #     conn._config[
    #         "sync_request_timeout"
    #     ] = timeout  # seconds. May need to be longer for real debugging.
    #     conn._config["allow_public_attrs"] = True
    #     conn._config["allow_pickle"] = True
    #     # This assumes the code is already synced over to the remote container
    #     remote_fn = getattr(conn.modules[self.fn.__module__], self.fn.__name__)
    #
    #     with redirected_stdio(conn):
    #         res = remote_fn(*args, **kwargs)
    #     conn.close()
    #     return res

    @property
    def config_for_rns(self):
        # TODO save Package resource, because fn_pointers are meaningless without the package.

        config = super().config_for_rns

        config.update(
            {
                "system": self._resource_string_for_subconfig(self.system),
                "env": self._resource_string_for_subconfig(self.env),
                "fn_pointers": self.fn_pointers,
                "resources": self.resources,
            }
        )
        return config

    def _save_sub_resources(self):
        if isinstance(self.system, Resource):
            self.system.save()

    # TODO maybe reuse these if we starting putting each function in its own container
    # @staticmethod
    # def run_ssh_cmd_in_cluster(ssh_key, ssh_user, address, cmd, port_fwd=None):
    #     subprocess.run("ssh -tt -o IdentitiesOnly=yes -i "
    #                    f"{ssh_key} {port_fwd or ''}"
    #                    f"{ssh_user}@{address} docker exec -it ray_container /bin/bash -c {cmd}".split(' '))

    def ssh(self):
        """SSH into the system associated with the function.

        Example:
            >>> remote_fn = rh.function(local_fn).to(gpu)
            >>> # SSH into gpu
            >>> remote_fn.ssh()
        """
        if self.system is None:
            raise RuntimeError("System must be specified and up to ssh into a Function")
        self.system.ssh()

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
        # Roughly trying to follow:
        # https://towardsdatascience.com/using-jupyter-notebook-running-on-a-remote-docker-container-via-ssh-ea2c3ebb9055
        # https://docs.ray.io/en/latest/ray-core/using-ray-with-jupyter.html
        if self.system is None:
            raise RuntimeError("Cannot SSH, running locally")

        tunnel, port_fwd = self.system.ssh_tunnel(
            local_port=port_forward, num_ports_to_try=10
        )
        try:
            install_cmd = "pip install jupyterlab"
            jupyter_cmd = f"jupyter lab --port {port_fwd} --no-browser"
            # port_fwd = '-L localhost:8888:localhost:8888 '  # TOOD may need when we add docker support
            with self.system.pause_autostop():
                self.system.run(commands=[install_cmd, jupyter_cmd], stream_logs=True)

        finally:
            if sync_package_on_close:
                if sync_package_on_close == "./":
                    sync_package_on_close = rh_config.rns_client.locate_working_dir()
                pkg = Package.from_string("local:" + sync_package_on_close)
                self.system._rsync(
                    source=f"~/{pkg.name}", dest=pkg.local_path, up=False
                )
            if not persist:
                tunnel.stop()
                kill_jupyter_cmd = f"jupyter notebook stop {port_fwd}"
                self.system.run(commands=[kill_jupyter_cmd])

    def get_or_call(self, run_name: str = None, *args, **kwargs) -> Any:
        """Check if Run was already completed, and if so return the result.
        If no cached Run is found on the cluster, create a new one and run it synchronously before
        returning its result.

        Args:
            run_name (Optional[str]): Name of a particular run for this function.
                If not provided will use the function's name.
            *args: Arguments to pass to the function for the run (relevant if creating a new run).
            **kwargs: Keyword arguments to pass to the function for the run (relevant if creating a new run).

        Returns:
            Any: Result of the Run

        Example:
            >>> # previously, remote_fn.run(arg1, arg2, run_name="my_async_run")
            >>> remote_fn.get_or_call()
        """
        from runhouse import Run

        run_name = run_name or Run._create_new_run_name(self.name)

        res = self._call_fn_with_ssh_access(
            fn_type="get_or_call", run_name=run_name, args=args, kwargs=kwargs
        )

        return res

    def get_or_run(self, run_name: str = None, *args, **kwargs) -> "Run":
        """Check if Run was already completed. If no cached Run is found on the cluster, create a new one.

        Note: If the Run has already completed, will not trigger a new Run.

        Args:
            run_name (Optional[str]): Name of a particular run for this function.
                If not provided will use the function's name.
            *args: Arguments to pass to the function for the run (relevant if creating a new run).
            **kwargs: Keyword arguments to pass to the function for the run (relevant if creating a new run).

        Returns:
            Run: Run object

        Example:
            >>> # previously, remote_fn.run(arg1, arg2, run_name="my_async_run")
            >>> remote_fn.get_or_call()
        """
        from runhouse import Run

        run_name = run_name or Run._create_new_run_name(self.name)
        if run_name == "latest":
            # TODO [JL]
            raise NotImplementedError("Latest not currently supported")

        completed_run: "Run" = self._call_fn_with_ssh_access(
            fn_type="get_or_run", run_name=run_name, args=args, kwargs=kwargs
        )

        return completed_run

    def keep_warm(
        self,
        autostop_mins=None,
        # TODO regions: List[str] = None,
        # TODO min_replicas: List[int] = None,
        # TODO max_replicas: List[int] = None
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
            # TODO put this in the current folder instead?
            module_path = Path.cwd() / (f"{name}_fn.py" if name else "sent_fn.py")
            logging.info(
                f"Writing out function function to {str(module_path)}. Please make "
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
            # from importlib.util import spec_from_file_location, module_from_spec
            # spec = spec_from_file_location(config['name'], str(module_path))
            # module = module_from_spec(spec)
            # spec.loader.exec_module(module)
            # new_fn = getattr(module, fn_pointers[2])
            # fn_pointers = Function._extract_fn_paths(raw_fn=new_fn, reqs=config['reqs'])


def function(
    fn: Optional[Union[str, Callable]] = None,
    name: Optional[str] = None,
    system: Optional[Union[str, Cluster]] = None,
    env: Optional[Union[List[str], Env, str]] = None,
    resources: Optional[dict] = None,
    dryrun: bool = False,
    load_secrets: bool = False,
    serialize_notebook_fn: bool = False,
    # args below are deprecated
    reqs: Optional[List[str]] = None,
    setup_cmds: Optional[List[str]] = None,
):
    """Builds an instance of :class:`Function`.

    Args:
        fn (Optional[str or Callable]): The function to execute on the remote system when the function is called.
        name (Optional[str]): Name of the Function to create or retrieve.
            This can be either from a local config or from the RNS.
        system (Optional[str or Cluster]): Hardware (cluster) on which to execute the Function.
            This can be either the string name of a Cluster object, or a Cluster object.
        env (Optional[List[str] or Env or str]): List of requirements to install on the remote cluster, or path to the
            requirements.txt file, or Env object or string name of an Env object.
        resources (Optional[dict]): Optional number (int) of resources needed to run the Function on the Cluster.
            Keys must be ``num_cpus`` and ``num_gpus``.
        dryrun (bool): Whether to create the Function if it doesn't exist, or load the Function object as a dryrun.
            (Default: ``False``)
        load_secrets (bool): Whether or not to send secrets; only applicable if `dryrun` is set to ``False``.
            (Default: ``False``)
        serialize_notebook_fn (bool): If function is of a notebook setting, whether or not to serialized the function.
            (Default: ``False``)

    Returns:
        Function: The resulting Function object.

    Example:
        >>> import runhouse as rh

        >>> cluster = rh.cluster(name="my_cluster")
        >>> def sum(a, b):
        >>>    return a + b

        >>> summer = rh.function(fn=sum, name="my_func").to(cluster, env=['requirements.txt']).save()

        >>> # using the function
        >>> res = summer(5, 8)  # returns 13

        >>> # Load function from above
        >>> reloaded_function = rh.function(name="my_func")
    """
    if name and not any([fn, system, env, resources]):
        # Try reloading existing function
        return Function.from_name(name, dryrun)

    if setup_cmds:
        warnings.warn(
            "``setup_cmds`` argument has been deprecated. "
            "Please pass in setup commands to rh.Env corresponding to the function instead."
        )
    if reqs is not None:
        warnings.warn(
            "``reqs`` argument has been deprecated. Please use ``env`` instead."
        )
        env = Env(reqs=reqs, setup_cmds=setup_cmds, working_dir="./")
    elif not isinstance(env, Env):
        env = _get_env_from(env) or Env()
        env.working_dir = env.working_dir or "./"

    fn_pointers = None
    if callable(fn):
        fn_pointers = Function._extract_fn_paths(raw_fn=fn, reqs=env.reqs)
        if fn_pointers[1] == "notebook":
            fn_pointers = Function._handle_nb_fn(
                fn,
                fn_pointers=fn_pointers,
                serialize_notebook_fn=serialize_notebook_fn,
                name=fn_pointers[2] or name,
            )
    elif isinstance(fn, str):
        # Url must match a regex of the form
        # 'https://github.com/username/repo_name/blob/branch_name/path/to/file.py:func_name'
        # Use a regex to extract username, repo_name, branch_name, path/to/file.py, and func_name
        pattern = (
            r"https://github\.com/(?P<username>[^/]+)/(?P<repo_name>[^/]+)/blob/"
            r"(?P<branch_name>[^/]+)/(?P<path>[^:]+):(?P<func_name>.+)"
        )
        match = re.match(pattern, fn)

        if match:
            username = match.group("username")
            repo_name = match.group("repo_name")
            branch_name = match.group("branch_name")
            path = match.group("path")
            func_name = match.group("func_name")
        else:
            raise ValueError(
                "fn must be a callable or string of the form "
                '"https://github.com/username/repo_name/blob/branch_name/path/to/file.py:func_name"'
            )
        module_name = Path(path).stem
        relative_path = str(repo_name / Path(path).parent)
        fn_pointers = (relative_path, module_name, func_name)
        # TODO [DG] check if the user already added this in their reqs
        repo_package = git_package(
            git_url=f"https://github.com/{username}/{repo_name}.git",
            revision=branch_name,
        )
        env.reqs.insert(0, repo_package)

    system = _get_cluster_from(system)

    new_function = Function(
        fn_pointers=fn_pointers,
        system=system,
        env=env,
        resources=resources,
        access=Function.DEFAULT_ACCESS,
        name=name,
        dryrun=dryrun,
    )

    if load_secrets and not dryrun:
        new_function.send_secrets()

    return new_function
