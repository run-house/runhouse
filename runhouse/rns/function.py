import copy
import inspect
import json
import logging
import os
import re
import warnings
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import requests

from runhouse import rh_config
from runhouse.rns.api_utils.resource_access import ResourceAccess
from runhouse.rns.api_utils.utils import is_jsonable, load_resp_content, read_resp_data
from runhouse.rns.envs import CondaEnv, Env
from runhouse.rns.hardware import Cluster
from runhouse.rns.packages import git_package, Package

from runhouse.rns.resource import Resource
from runhouse.rns.run_module_utils import call_fn_by_type

from runhouse.rns.utils.env import _get_env_from

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
        serialize_notebook_fn=False,
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
        self.serialize_notebook_fn = serialize_notebook_fn
        self.access = access or self.DEFAULT_ACCESS
        self.dryrun = dryrun
        self.resources = resources or {}
        super().__init__(name=name, dryrun=dryrun)

        if not self.dryrun:
            self = self.to(self.system, env=self.env)

    # ----------------- Constructor helper methods -----------------

    @staticmethod
    def from_config(config: dict, dryrun: bool = True):
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
        if isinstance(system, str) and rh_config.rns_client.exists(system):
            # if the system is set to a cluster
            cluster_config: dict = rh_config.rns_client.load_config(name=system)
            if not cluster_config:
                raise Exception(f"No cluster config saved for {system}")

            # set the cluster config as the system
            config["system"] = cluster_config
        return config

    def to(
        self,
        system: Union[str, Cluster] = None,
        env: Union[List[str], Env] = None,
        # Variables below are deprecated
        reqs: Optional[List[str]] = None,
        setup_cmds: Optional[List[str]] = [],
    ):
        """
        Set up a Function and Env on the given system.

        See the args of the factory method :func:`function` for more information.
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
            env = env or self.env
            env = _get_env_from(env)

        if self.env:
            # Note: Here we add the existing reqs in the function’s env into the new env
            # (otherwise we don’t have a way to add in "./")
            new_reqs = [req for req in self.env.reqs if req not in env.reqs]
            env.reqs += new_reqs

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

    def run_setup(self, cmds: List[str]):
        """Run the given setup commands on the system."""
        self.system.run(cmds)

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

    def __call__(self, *args, stream_logs=False, **kwargs):
        fn_type = "call"
        if self.access in [ResourceAccess.WRITE, ResourceAccess.READ]:
            if not self.system or self.system.name == rh_config.obj_store.cluster_name:
                [relative_path, module_name, fn_name] = self.fn_pointers
                conda_env = (
                    self.env.env_name
                    if self.env and isinstance(self.env, CondaEnv)
                    else None
                )
                # If we're on this cluster, don't pickle the result before passing back.
                # We need to pickle before passing back in most cases because the env in
                # which the function executes may have a different set of packages than the
                # server, so when Ray passes a result back into the server it will may fail to
                # unpickle. We assume the user's client has the necessary packages to unpickle
                # their own result.
                serialize_res = not self.system.on_this_cluster()
                return call_fn_by_type(
                    fn_type=fn_type,
                    fn_name=fn_name,
                    relative_path=relative_path,
                    module_name=module_name,
                    resources=self.resources,
                    conda_env=conda_env,
                    args=args,
                    kwargs=kwargs,
                    serialize_res=serialize_res,
                )
            elif stream_logs:
                run_key = self.remote(*args, **kwargs)
                return self.system.get(run_key, stream_logs=True)
            else:
                return self._call_fn_with_ssh_access(
                    fn_type=fn_type, args=args, kwargs=kwargs
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

            res = read_resp_data(resp)
            return res

    def repeat(self, num_repeats: int, *args, **kwargs):
        """Repeat the Function call multiple times.

        Args:
            num_repeats (int): Number of times to repeat the Function call.
            *args: Positional arguments to pass to the Function
            **kwargs: Keyword arguments to pass to the Function
        """
        if self.access in [ResourceAccess.WRITE, ResourceAccess.READ]:
            return self._call_fn_with_ssh_access(
                fn_type="repeat", args=[num_repeats, args], kwargs=kwargs
            )
        else:
            raise NotImplementedError(
                "Function.repeat only works with Write or Read access, not Proxy access"
            )

    def map(self, arg_list, **kwargs):
        """Map a function over a list of arguments."""
        if self.access in [ResourceAccess.WRITE, ResourceAccess.READ]:
            return self._call_fn_with_ssh_access(
                fn_type="map", args=arg_list, kwargs=kwargs
            )
        else:
            raise NotImplementedError(
                "Function.map only works with Write or Read access, not Proxy access"
            )

    def starmap(self, args_lists, **kwargs):
        """Like :func:`map` except that the elements of the iterable are expected to be iterables
        that are unpacked as arguments. An iterable of [(1,2), (3, 4)] results in [func(1,2), func(3,4)]."""
        if self.access in [ResourceAccess.WRITE, ResourceAccess.READ]:
            return self._call_fn_with_ssh_access(
                fn_type="starmap", args=args_lists, kwargs=kwargs
            )
        else:
            raise NotImplementedError(
                "Function.starmap only works with Write or Read access, not Proxy access"
            )

    def enqueue(self, resources: Optional[Dict] = None, *args, **kwargs):
        """Enqueue a Function call to be run later."""
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
        """Run async remote call on cluster."""
        # TODO [DG] pin the run_key and return a string (printed to log) so result can be retrieved later and we
        # don't need to init ray here. Also, allow user to pass the string as a param to remote().
        # TODO [DG] add rpc for listing gettaable strings, plus metadata (e.g. when it was created)
        # We need to ray init here so the returned Ray object ref doesn't throw an error it's deserialized
        # import ray
        # ray.init(ignore_reinit_error=True)
        if self.access in [ResourceAccess.WRITE, ResourceAccess.READ]:
            run_key = self._call_fn_with_ssh_access(
                fn_type="remote", args=args, kwargs=kwargs
            )
            cluster_name = (
                f'rh.cluster(name="{self.system.rns_address}")'
                if self.system.name
                else "<my_cluster>"
            )
            # TODO print this nicely
            logger.info(
                f"Submitted remote call to cluster. Result or logs can be retrieved"
                f'\n with run_key "{run_key}", e.g. '
                f'\n`{cluster_name}.get("{run_key}", stream_logs=True)` in python '
                f'\n`runhouse logs "{self.system.name}" {run_key}` from the command line.'
                f"\n or cancelled with "
                f'\n`{cluster_name}.cancel("{run_key}")` in python or '
                f'\n`runhouse cancel "{self.system.name}" {run_key}` from the command line.'
            )
            return run_key
        else:
            raise NotImplementedError(
                "Function.remote only works with Write or Read access, not Proxy access"
            )

    def get(self, run_key):
        """Get the result of a Function call that was submitted as async using `remote`.

        Args:
            run_key: A single or list of runhouse run_key strings returned by a Function.remote() call. The ObjectRefs
                must be from the cluster that this Function is running on.
        """
        return self.system.get(run_key)

    def _call_fn_with_ssh_access(self, fn_type, resources=None, args=None, kwargs=None):
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
        res = self.system.run_module(
            relative_path,
            module_name,
            fn_name,
            fn_type,
            resources,
            env_name,
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
        self.system.save()

    # TODO maybe reuse these if we starting putting each function in its own container
    # @staticmethod
    # def run_ssh_cmd_in_cluster(ssh_key, ssh_user, address, cmd, port_fwd=None):
    #     subprocess.run("ssh -tt -o IdentitiesOnly=yes -i "
    #                    f"{ssh_key} {port_fwd or ''}"
    #                    f"{ssh_user}@{address} docker exec -it ray_container /bin/bash -c {cmd}".split(' '))

    def ssh(self):
        """SSH into the system."""
        if self.system is None:
            raise RuntimeError("System must be specified and up to ssh into a Function")
        self.system.ssh()

    def send_secrets(self, providers: Optional[List[str]] = None):
        """Send secrets to the system."""
        self.system.send_secrets(providers=providers)

    def http_url(self, curl_command=False, *args, **kwargs) -> str:
        """
        Return the endpoint needed to run the Function on the remote cluster, or provide the curl command if requested.
        """
        raise NotImplementedError("http_url not yet implemented for Function")
        resource_uri = rh_config.rns_client.resource_uri(name=self.name)
        uri = f"proxy/{resource_uri}"
        if curl_command:
            # NOTE: curl command should include args and kwargs - this will help us generate better API docs
            if not is_jsonable(args) or not is_jsonable(kwargs):
                raise Exception(
                    "Invalid Function func params provided, must be able to convert args and kwargs to json"
                )

            return (
                "curl -X 'POST' '{api_server_url}/proxy{resource_uri}/endpoint' "
                "-H 'accept: application/json' "
                "-H 'Authorization: Bearer {auth_token}' "
                "-H 'Content-Type: application/json' "
                "-d '{data}'".format(
                    api_server_url=rh_config.rns_client.api_server_url,
                    resource_uri=uri,
                    auth_token=rh_config.rns_client.token,
                    data=json.dumps({"args": args, "kwargs": kwargs}),
                )
            )

        # HTTP URL needed to run the Function remotely
        http_url = f"{rh_config.rns_client.api_server_url}/{uri}/endpoint"
        return http_url

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
                self.system.rsync(source=f"~/{pkg.name}", dest=pkg.local_path, up=False)
            if not persist:
                tunnel.stop()
                kill_jupyter_cmd = f"jupyter notebook stop {port_fwd}"
                self.system.run(commands=[kill_jupyter_cmd])

    def keep_warm(
        self,
        autostop_mins=None,
        # TODO regions: List[str] = None,
        # TODO min_replicas: List[int] = None,
        # TODO max_replicas: List[int] = None
    ):
        """Keep the system warm for autostop_mins. If autostop_mins is ``None`` or -1, keep warm indefinitely."""
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
    env: Union[Optional[List[str]], Env] = None,
    resources: Optional[dict] = None,
    # TODO image: Optional[str] = None,
    dryrun: bool = False,
    load_secrets: bool = False,
    serialize_notebook_fn: bool = False,
    load: bool = True,
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
        env (Optional[List[str]]): List of requirements to install on the remote cluster, or path to the
            requirements.txt file.
        resources (Optional[dict]): Optional number (int) of resources needed to run the Function on the Cluster.
            Keys must be ``num_cpus`` and ``num_gpus``.
        dryrun (bool): Whether to create the Function if it doesn't exist, or load the Function object as a dryrun.
            (Default: ``False``)
        load_secrets (bool): Whether or not to send secrets; only applicable if `dryrun` is set to ``False``.
            (Default: ``False``)
        serialize_notebook_fn (bool): If function is of a notebook setting, whether or not to serialized the function.
            (Default: ``False``)
        load (bool): Whether to load an existing config for the Function. (Default: ``True``)

    Returns:
        Function: The resulting Function object.

    Example:
        >>> def sum(a, b):
        >>>    return a + b
        >>>
        >>> # creating the function
        >>> summer = rh.function(fn=sum, system=cluster, env=['requirements.txt'])
        >>> # or, equivalently
        >>> summer = rh.function(fn=sum).to(cluster, env=['requirements.txt'])
        >>>
        >>> # using the function
        >>> summer(5, 8)  # returns 13
    """

    config = rh_config.rns_client.load_config(name) if load else {}
    config["name"] = name or config.get("rns_address", None) or config.get("name")
    config["resources"] = (
        resources if resources is not None else config.get("resources")
    )
    config["serialize_notebook_fn"] = serialize_notebook_fn or config.get(
        "serialize_notebook_fn"
    )

    if setup_cmds:
        warnings.warn(
            "``setup_cmds`` argument has been deprecated. "
            "Please pass in setup commands to rh.Env corresponding to the function instead."
        )
    if reqs is not None:
        warnings.warn(
            "``reqs`` argument has been deprecated. Please use ``env`` instead."
        )
        env = Env(reqs=reqs, setup_cmds=setup_cmds)
    else:
        env = env or config.get("env")
        env = _get_env_from(env)

    reqs = env.reqs if env else []
    if callable(fn):
        if not [
            req
            for req in reqs
            if (isinstance(req, str) and "./" in req)
            or (isinstance(req, Package) and req.is_local())
        ]:
            reqs.append("./")
        fn_pointers = Function._extract_fn_paths(raw_fn=fn, reqs=reqs)
        if fn_pointers[1] == "notebook":
            fn_pointers = Function._handle_nb_fn(
                fn,
                fn_pointers=fn_pointers,
                serialize_notebook_fn=serialize_notebook_fn,
                name=fn_pointers[2] or config["name"],
            )
        config["fn_pointers"] = fn_pointers
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
        config["fn_pointers"] = (relative_path, module_name, func_name)
        # TODO [DG] check if the user already added this in their reqs
        repo_package = git_package(
            git_url=f"https://github.com/{username}/{repo_name}.git",
            revision=branch_name,
        )
        reqs.insert(0, repo_package)

    if env:
        env.reqs = reqs
    else:
        env = Env(reqs=reqs, setup_cmds=setup_cmds)

    config["env"] = env
    config["system"] = system or config.get("system")
    if isinstance(config["system"], str):
        hw_dict = rh_config.rns_client.load_config(config["system"])
        if not hw_dict:
            raise RuntimeError(
                f'Cluster {rh_config.rns_client.resolve_rns_path(config["system"])} '
                f"not found locally or in RNS."
            )
        config["system"] = hw_dict

    config["access_level"] = config.get("access_level", Function.DEFAULT_ACCESS)

    new_function = Function.from_config(config, dryrun=dryrun)

    if load_secrets and not dryrun:
        new_function.send_secrets()

    return new_function
