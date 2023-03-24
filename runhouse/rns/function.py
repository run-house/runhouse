import copy
import inspect
import json
import logging
import os
import re
from pathlib import Path
from typing import Callable, List, Optional, Tuple, Union

import requests

from runhouse import rh_config
from runhouse.rns.api_utils.resource_access import ResourceAccess
from runhouse.rns.api_utils.utils import is_jsonable, load_resp_content, read_resp_data
from runhouse.rns.hardware import Cluster
from runhouse.rns.packages import git_package, Package

from runhouse.rns.resource import Resource
from runhouse.rns.run_module_utils import call_fn_by_type, get_fn_by_name

logger = logging.getLogger(__name__)


class Function(Resource):
    RESOURCE_TYPE = "function"
    DEFAULT_ACCESS = "write"

    def __init__(
        self,
        fn_pointers: Tuple[str, str, str],
        system: Optional[Cluster] = None,
        name: Optional[str] = None,
        reqs: Optional[List[str]] = None,
        setup_cmds: Optional[List[str]] = None,
        dryrun: bool = False,
        access: Optional[str] = None,
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
        self.reqs = reqs
        self.setup_cmds = setup_cmds or []
        self.access = access or self.DEFAULT_ACCESS
        self.dryrun = dryrun
        super().__init__(name=name, dryrun=dryrun)

        if not self.dryrun and self.system and self.access in ["write", "read"]:
            self.to(self.system, reqs=self.reqs, setup_cmds=setup_cmds)

    # ----------------- Constructor helper methods -----------------

    @staticmethod
    def from_config(config: dict, dryrun: bool = True):
        """Create a Function object from a config dictionary."""
        config["reqs"] = [
            Package.from_config(package, dryrun=True)
            if isinstance(package, dict)
            else package
            for package in config["reqs"]
        ]

        if isinstance(config["system"], dict):
            config["system"] = Cluster.from_config(config["system"], dryrun=dryrun)

        if "fn_pointers" not in config:
            raise ValueError(
                "No fn_pointers provided in config. Please provide a path "
                "to a python file, module, and function name."
            )

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
        system: Union[str, Cluster],
        reqs: Optional[List[str]] = None,
        setup_cmds: Optional[List[str]] = None,
    ):
        """
        Set up a Function on the given system, install the reqs, and run setup_cmds.

        See the args of the factory method :func:`function` for more information.
        """
        # We need to backup the system here so the __getstate__ method of the cluster
        # doesn't wipe the client and _grpc_client of this function's cluster when
        # deepcopy copies it.
        hw_backup = self.system
        self.system = None
        new_function = copy.deepcopy(self)
        self.system = hw_backup
        if isinstance(system, str):
            system = Cluster.from_name(system, dryrun=self.dryrun)
        new_function.system = system
        new_function.reqs = reqs if reqs else self.reqs
        new_function.setup_cmds = setup_cmds if setup_cmds else self.setup_cmds

        logging.info("Setting up Function on cluster.")
        new_function.system.install_packages(new_function.reqs)
        if self.setup_cmds:
            new_function.system.run(self.setup_cmds)
        logging.info("Function setup complete.")

        return new_function

    def run_setup(self, cmds: List[str], force: bool = False):
        """Run the given setup commands on the system."""
        to_run = []
        for cmd in cmds:
            if force or cmd not in self.setup_cmds:
                to_run.append(cmd)
        if to_run:
            self.setup_cmds.extend(to_run)
            self.system.run(to_run)

    @staticmethod
    def extract_fn_paths(raw_fn: Callable, reqs: List[str]):
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
            if not isinstance(req, str) and req.is_local():
                local_path = Path(req.local_path)
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
                fn = get_fn_by_name(
                    module_name=module_name,
                    fn_name=fn_name,
                    relative_path=relative_path,
                )
                return call_fn_by_type(
                    fn, fn_type, fn_name, relative_path, args, kwargs
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

    def enqueue(self, *args, **kwargs):
        """Enqueue a Function call to be run later."""
        if self.access in [ResourceAccess.WRITE, ResourceAccess.READ]:
            return self._call_fn_with_ssh_access(
                fn_type="queue", args=args, kwargs=kwargs
            )
        else:
            raise NotImplementedError(
                "Function.enqueue only works with Write or Read access, not Proxy access"
            )

    def remote(self, *args, **kwargs):
        """Run async remote call on cluster."""
        # TODO [DG] pin the obj_ref and return a string (printed to log) so result can be retrieved later and we
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

    def get(self, obj_ref):
        """Get the result of a Function call that was submitted as async using `remote`.

        Args:
            obj_ref: A single or list of Ray.ObjectRef objects returned by a Function.remote() call. The ObjectRefs
                must be from the cluster that this Function is running on.
        """
        # TODO [DG] replace with self.system.get()?
        if self.access in [ResourceAccess.WRITE, ResourceAccess.READ]:
            arg_list = obj_ref if isinstance(obj_ref, list) else [obj_ref]
            return self._call_fn_with_ssh_access(
                fn_type="get", args=arg_list, kwargs={}
            )
        else:
            raise NotImplementedError(
                "Function.get only works with Write or Read access, not Proxy access"
            )

    def _call_fn_with_ssh_access(self, fn_type, args, kwargs):
        # https://docs.ray.io/en/latest/ray-core/tasks/patterns/map-reduce.html
        # return ray.get([map.remote(i, map_func) for i in replicas])
        # TODO allow specifying resources per worker for map
        # TODO [DG] check whether we're on the cluster and if so, just call the function directly via the
        # helper function currently in UnaryServer
        name = self.name or "anonymous function"
        if self.fn_pointers is None:
            raise RuntimeError(f"No fn pointers saved for {name}")

        [relative_path, module_name, fn_name] = self.fn_pointers
        name = self.name or fn_name or "anonymous function"
        logger.info(f"Running {name} via gRPC")
        res = self.system.run_module(
            relative_path, module_name, fn_name, fn_type, args, kwargs
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
                "reqs": [
                    self._resource_string_for_subconfig(package)
                    for package in self.reqs
                ],
                "setup_cmds": self.setup_cmds,
                "fn_pointers": self.fn_pointers,
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
            # This will all be cloudpickled by the gRPC client and unpickled by the gRPC server
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
            # fn_pointers = Function.extract_fn_paths(raw_fn=new_fn, reqs=config['reqs'])


def function(
    fn: Optional[Union[str, Callable]] = None,
    name: Optional[str] = None,
    system: Optional[Union[str, Cluster]] = None,
    reqs: Optional[List[str]] = None,
    setup_cmds: Optional[List[str]] = None,
    # TODO image: Optional[str] = None,
    dryrun: bool = False,
    load_secrets: bool = False,
    serialize_notebook_fn: bool = False,
    load: bool = True,
):
    """Factory method for constructing a Runhouse Function object.

    Args:
        fn (Optional[str or Callable]): The function to execute on the remote system when the function is called.
        name (Optional[str]): Name of the Function to create or retrieve.
            This can be either from a local config or from the RNS.
        system (Optional[str or Cluster]): Hardware (cluster) on which to execute the Function.
            This can be either the string name of a Cluster object, or a Cluster object.
        reqs (Optional[List[str]]): List of requirements to install on the remote cluster, or path to the
            requirements.txt file.
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
        >>> summer = rh.function(fn=sum, system=cluster, reqs=['requirements.txt'])
        >>> # or, equivalently
        >>> summer = rh.function(fn=sum).to(cluster, reqs=['requirements.txt'])
        >>>
        >>> # using the function
        >>> summer(5, 8)  # returns 13
    """

    config = rh_config.rns_client.load_config(name) if load else {}
    config["name"] = name or config.get("rns_address", None) or config.get("name")
    config["reqs"] = reqs if reqs is not None else config.get("reqs", [])

    processed_reqs = []
    for req in config["reqs"]:
        # TODO [DG] the following is wrong. RNS address doesn't have to start with '/'. However if we check if each
        #  string exists in RNS this will be incredibly slow, so leave it for now.
        if isinstance(req, str) and req[0] == "/" and rh_config.rns_client.exists(req):
            # If req is an rns address
            req = rh_config.rns_client.load_config(req)
        processed_reqs.append(req)
    config["reqs"] = processed_reqs

    if callable(fn):
        if not [req for req in config["reqs"] if "./" in req]:
            config["reqs"].append("./")
        fn_pointers = Function.extract_fn_paths(raw_fn=fn, reqs=config["reqs"])
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
        config["reqs"].insert(0, repo_package)

    config["system"] = system or config.get("system")
    if isinstance(config["system"], str):
        hw_dict = rh_config.rns_client.load_config(config["system"])
        if not hw_dict:
            raise RuntimeError(
                f'Cluster {rh_config.rns_client.resolve_rns_path(config["system"])} '
                f"not found locally or in RNS."
            )
        config["system"] = hw_dict

    config["setup_cmds"] = (
        setup_cmds if setup_cmds is not None else config.get("setup_cmds")
    )

    config["access_level"] = config.get("access_level", Function.DEFAULT_ACCESS)

    new_function = Function.from_config(config, dryrun=dryrun)

    if load_secrets and not dryrun:
        new_function.send_secrets()

    return new_function
