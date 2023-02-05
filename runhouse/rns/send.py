import copy
import inspect
import json
import logging
import os
import re
from pathlib import Path
from typing import Callable, List, Optional, Tuple, Union

import grpc
import requests
import sshtunnel

from runhouse import rh_config
from runhouse.rns.api_utils.resource_access import ResourceAccess
from runhouse.rns.api_utils.utils import is_jsonable, read_response_data
from runhouse.rns.hardware import Cluster
from runhouse.rns.packages import git_package, Package

from runhouse.rns.resource import Resource
from runhouse.rns.run_module_utils import call_fn_by_type, get_fn_by_name

logger = logging.getLogger(__name__)


class Send(Resource):
    RESOURCE_TYPE = "send"
    DEFAULT_ACCESS = "write"

    def __init__(
        self,
        fn_pointers: Tuple[str, str, str],
        hardware: Optional[Cluster] = None,
        name: [Optional[str]] = None,
        reqs: Optional[List[str]] = None,
        setup_cmds: Optional[List[str]] = None,
        dryrun: bool = True,
        access: Optional[str] = None,
        **kwargs,  # We have this here to ignore extra arguments when calling from from_config
    ):
        """
        Create, load, or update a Send ("Serverless endpoint"). A Send is comprised of the
        entrypoint, hardware, and dependencies to run the service.

        Args:
            fn (): A python callable or entrypoint string (module:function) within the package which the user
                can call to run remotely as a microservice. The Send object is callable, taking the same inputs
                 and producing the same outputs as fn. For example, if we create
                 `my_send = Send(fn=lambda x: x+1, hardware=my_hw)`, we can call it with `my_send(4)`, and
                 the fn will run remotely on `my_hw`.
            name (): A URI to persist this Send's metadata to Runhouse's Resource Naming System (RNS), which allows
                it to be reloaded and used later or in other environments. If name is left empty, the Send is
                "anonymous" and will only exist within this Python context. # TODO more about user namespaces.
            hardware ():
            reqs ():
            cluster ():
        """
        self.fn_pointers = fn_pointers
        self.hardware = hardware
        self.reqs = reqs
        self.setup_cmds = setup_cmds or []
        self.access = access or self.DEFAULT_ACCESS
        self.dryrun = dryrun
        super().__init__(name=name, dryrun=dryrun)

        if not self.dryrun and self.hardware and self.access in ["write", "read"]:
            self.to(self.hardware, reqs=self.reqs, setup_cmds=setup_cmds)

    # ----------------- Constructor helper methods -----------------

    @staticmethod
    def from_config(config, dryrun=True):
        """Create a Send object from a config dictionary.

        Args:
            config (dict): Dictionary of config values.

        Returns:
            Send: Send object created from config values.
        """
        config["reqs"] = [
            Package.from_config(package, dryrun=True)
            if isinstance(package, dict)
            else package
            for package in config["reqs"]
        ]
        # TODO validate which fields need to be present in the config

        if isinstance(config["hardware"], dict):
            config["hardware"] = Cluster.from_config(config["hardware"], dryrun=dryrun)

        if "fn_pointers" not in config:
            raise ValueError(
                "No fn_pointers provided in config. Please provide a path "
                "to a python file, module, and function name."
            )

        return Send(**config, dryrun=dryrun)

    def to(self, hardware, reqs=None, setup_cmds=None):
        new_send = copy.deepcopy(self)
        new_send.hardware = hardware if hardware else self.hardware
        new_send.reqs = reqs if reqs else self.reqs
        new_send.setup_cmds = (
            setup_cmds if setup_cmds else self.setup_cmds
        )  # Run inside reup_cluster
        # TODO [DG] figure out how to run setup_cmds on BYO Cluster

        logging.info("Setting up Send on cluster.")
        if not new_send.hardware.address:
            # For SkyCluster, this initial check doesn't trigger a sky.status, which is slow.
            # If cluster simply doesn't have an address we likely need to up it.
            if not hasattr(new_send.hardware, "up"):
                raise ValueError(
                    "Cluster must have an address (i.e. be up) or have a reup_cluster method "
                    "(e.g. SkyCluster)."
                )
            if not new_send.hardware.is_up():
                # If this is a SkyCluster, before we up the cluster, run a sky.check to see if the cluster
                # is already up but doesn't have an address assigned yet.
                new_send.reup_cluster()
        try:
            new_send.hardware.install_packages(new_send.reqs)
        except (grpc.RpcError, sshtunnel.BaseSSHTunnelForwarderError):
            # It's possible that the cluster went down while we were trying to install packages.
            if not new_send.hardware.is_up():
                new_send.reup_cluster()
            else:
                new_send.hardware.restart_grpc_server(resync_rh=False)
            new_send.hardware.install_packages(new_send.reqs)
        logging.info("Send setup complete.")

        return new_send

    def reup_cluster(self):
        logger.info(f"Upping the cluster {self.hardware.name}")
        self.hardware.up()
        # TODO [DG] this only happens when the cluster comes up, not when a new send is added to the cluster
        self.hardware.run(self.setup_cmds)

    def run_setup(self, cmds, force=False):
        to_run = []
        for cmd in cmds:
            if force or cmd not in self.setup_cmds:
                to_run.append(cmd)
        if to_run:
            self.setup_cmds.extend(to_run)
            self.hardware.run(to_run)

    @staticmethod
    def extract_fn_paths(raw_fn, reqs):
        """Get the path to the module, module name, and function name to be able to import it on the server"""
        if not isinstance(raw_fn, Callable):
            raise TypeError(
                f"Invalid fn for Send, expected Callable but received {type(raw_fn)}"
            )
        # Background on all these dunders: https://docs.python.org/3/reference/import.html
        module = inspect.getmodule(raw_fn)

        # Need to resolve in case just filename is given
        module_path = (
            str(Path(inspect.getfile(module)).resolve())
            if hasattr(module, "__file__")
            else None
        )

        if not module_path or raw_fn.__name__ == "<lambda>":
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
            # module_name = getattr(module.__spec__, 'name', inspect.getmodulename(module_path))
            module_name = (
                module.__spec__.name
                if getattr(module, "__package__", False)
                else inspect.getmodulename(module_path)
            )
            # TODO __qualname__ doesn't work when fn is aliased funnily, like torch.sum
            fn_name = getattr(raw_fn, "__qualname__", raw_fn.__name__)

        # if module is not in a package, we need to add its parent directory to the path to import it
        # if not getattr(module, '__package__', None):
        #     module_path = os.path.dirname(module.__file__)

        remote_import_path = None
        for req in reqs:
            local_path = None
            if not isinstance(req, str) and req.is_local():
                local_path = Path(req.local_path)
            elif isinstance(req, str):
                if req.split(":")[0] in ["local", "reqs"]:
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
                    remote_import_path = (
                        local_path.name
                        + "/"
                        + str(Path(root_path).relative_to(local_path))
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

    # ----------------- Send call methods -----------------

    def __call__(self, *args, stream_logs=False, **kwargs):
        if self.access in [ResourceAccess.write, ResourceAccess.read]:
            if (
                not self.hardware
                or self.hardware.name == rh_config.obj_store.cluster_name
            ):
                fn = get_fn_by_name(
                    module_name=self.fn_pointers[1], fn_name=self.fn_pointers[2]
                )
                return call_fn_by_type(
                    fn, fn_type, fn_name, relative_path, args, kwargs
                )
            elif stream_logs:
                run_key = self.remote(*args, **kwargs)
                return self.hardware.get(run_key, stream_logs=True)
            else:
                return self._call_fn_with_ssh_access(
                    fn_type="call", args=args, kwargs=kwargs
                )
        else:
            # run the function via http url - user only needs Proxy access
            if self.access != ResourceAccess.proxy:
                raise RuntimeError("Running http url requires proxy access")
            if not rh_config.rns_client.token:
                raise ValueError(
                    "Token must be saved in the local .rh config in order to use an http url"
                )
            http_url = self.http_url()
            logger.info(f"Running {self.name} via http url: {http_url}")
            resp = requests.post(
                http_url,
                data=json.dumps({"args": args, "kwargs": kwargs}),
                headers=rh_config.rns_client.request_headers,
            )
            if resp.status_code != 200:
                raise Exception(
                    f"Failed to run Send endpoint: {json.loads(resp.content)}"
                )

            res = read_response_data(resp)
            return res

    def repeat(self, num_repeats, *args, **kwargs):
        """Repeat the Send call multiple times.

        Args:
            num_repeats (int): Number of times to repeat the Send call.
            *args: Positional arguments to pass to the Send
            **kwargs: Keyword arguments to pass to the Send
        """
        if self.access in [ResourceAccess.write, ResourceAccess.read]:
            return self._call_fn_with_ssh_access(
                fn_type="repeat", args=[num_repeats, args], kwargs=kwargs
            )
        else:
            raise NotImplementedError(
                "Send.repeat only works with Write or Read access, not Proxy access"
            )

    def map(self, arg_list, **kwargs):
        """Map a function over a list of arguments."""
        if self.access in [ResourceAccess.write, ResourceAccess.read]:
            return self._call_fn_with_ssh_access(
                fn_type="map", args=arg_list, kwargs=kwargs
            )
        else:
            raise NotImplementedError(
                "Send.map only works with Write or Read access, not Proxy access"
            )

    def starmap(self, args_lists, **kwargs):
        """Like Send.map() except that the elements of the iterable are expected to be iterables
        that are unpacked as arguments. An iterable of [(1,2), (3, 4)] results in [func(1,2), func(3,4)]."""
        if self.access in [ResourceAccess.write, ResourceAccess.read]:
            return self._call_fn_with_ssh_access(
                fn_type="starmap", args=args_lists, kwargs=kwargs
            )
        else:
            raise NotImplementedError(
                "Send.starmap only works with Write or Read access, not Proxy access"
            )

    def enqueue(self, *args, **kwargs):
        """Enqueue a Send call to be run later.

        Args:
            *args: Positional arguments to pass to the Send
            **kwargs: Keyword arguments to pass to the Send
        """
        if self.access in [ResourceAccess.write, ResourceAccess.read]:
            return self._call_fn_with_ssh_access(
                fn_type="queue", args=args, kwargs=kwargs
            )
        else:
            raise NotImplementedError(
                "Send.enqueue only works with Write or Read access, not Proxy access"
            )

    def remote(self, *args, **kwargs):
        """Map a function over a list of arguments."""
        # TODO [DG] pin the obj_ref and return a string (printed to log) so result can be retrieved later and we
        # don't need to init ray here. Also, allow user to pass the string as a param to remote().
        # TODO [DG] add rpc for listing gettaable strings, plus metadata (e.g. when it was created)
        # We need to ray init here so the returned Ray object ref doesn't throw an error it's deserialized
        # import ray
        # ray.init(ignore_reinit_error=True)
        if self.access in [ResourceAccess.write, ResourceAccess.read]:
            run_key = self._call_fn_with_ssh_access(
                fn_type="remote", args=args, kwargs=kwargs
            )
            cluster_name = (
                f'rh.cluster(name="{self.hardware.rns_address}")'
                if self.hardware.name
                else "<my_cluster>"
            )
            logger.info(
                f"Submitted remote call to cluster. Result or logs can be retrieved"
                f'\n with run_key "{run_key}", e.g. '
                f'\n`{cluster_name}.get("{run_key}", stream_logs=True)` in python '
                f"\n or cancelled with "
                f'\n`{cluster_name}.cancel("{run_key}")` in python or '
                f'\n`runhouse cancel "{cluster_name}" {run_key}` from the command line.'
            )
            return run_key
        else:
            raise NotImplementedError(
                "Send.remote only works with Write or Read access, not Proxy access"
            )

    def get(self, obj_ref):
        """Get the result of a Send call that was submitted as async using `remote`.

        Args:
            obj_ref: A single or list of Ray.ObjectRef objects returned by a Send.remote() call. The ObjectRefs
                must be from the cluster that this Send is running on.
        """
        # TODO [DG] replace with self.hardware.get()?
        if self.access in [ResourceAccess.write, ResourceAccess.read]:
            arg_list = obj_ref if isinstance(obj_ref, list) else [obj_ref]
            return self._call_fn_with_ssh_access(
                fn_type="get", args=arg_list, kwargs={}
            )
        else:
            raise NotImplementedError(
                "Send.get only works with Write or Read access, not Proxy access"
            )

    def _call_fn_with_ssh_access(self, fn_type, args, kwargs):
        # https://docs.ray.io/en/latest/ray-core/tasks/patterns/map-reduce.html
        # return ray.get([map.remote(i, map_func) for i in replicas])
        # TODO allow specifying resources per worker for map
        # TODO [DG] check whether we're on the cluster and if so, just call the function directly via the
        # helper function currently in UnaryServer
        name = self.name or "anonymous send"
        if self.fn_pointers is None:
            raise RuntimeError(f"No fn pointers saved for {name}")

        [relative_path, module_name, fn_name] = self.fn_pointers
        name = self.name or fn_name or "anonymous send"
        logger.info(f"Running {name} via gRPC")
        res = self.hardware.run_module(
            relative_path, module_name, fn_name, fn_type, args, kwargs
        )
        return res

    # TODO [DG] test this properly
    # def debug(self, redirect_logging=False, timeout=10000, *args, **kwargs):
    #     """Run the Send in debug mode. This will run the Send through a tunnel interpreter, which
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
    #     creds = self.hardware.ssh_creds()
    #     ssh_client = ParamikoMachine(
    #         self.hardware.address,
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
                "hardware": self._resource_string_for_subconfig(self.hardware.save()),
                "reqs": [
                    self._resource_string_for_subconfig(package)
                    for package in self.reqs
                ],
                "setup_cmds": self.setup_cmds,
                "fn_pointers": self.fn_pointers,
            }
        )
        return config

    # TODO maybe reuse these if we starting putting each send in its own container
    # @staticmethod
    # def run_ssh_cmd_in_cluster(ssh_key, ssh_user, address, cmd, port_fwd=None):
    #     subprocess.run("ssh -tt -o IdentitiesOnly=yes -i "
    #                    f"{ssh_key} {port_fwd or ''}"
    #                    f"{ssh_user}@{address} docker exec -it ray_container /bin/bash -c {cmd}".split(' '))

    def ssh(self):
        if self.hardware is None:
            raise RuntimeError("Hardware must be specified and up to ssh into a Send")
        self.hardware.ssh()

    def send_secrets(self, reload=False):
        self.hardware.send_secrets(reload=reload)

    def http_url(self, curl_command=False, *args, **kwargs) -> str:
        """Return the endpoint needed to run the Send on the remote cluster, or provide the curl command if requested"""
        resource_uri = rh_config.rns_client.resource_uri(name=self.name)
        uri = f"proxy/{resource_uri}"
        if curl_command:
            # NOTE: curl command should include args and kwargs - this will help us generate better API docs
            if not is_jsonable(args) or not is_jsonable(kwargs):
                raise Exception(
                    "Invalid Send func params provided, must be able to convert args and kwargs to json"
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

        # HTTP URL needed to run the Send remotely
        http_url = f"{rh_config.rns_client.api_server_url}/{uri}/endpoint"
        return http_url

    def notebook(self, persist=False, sync_package_on_close=None, port_forward=8888):
        # Roughly trying to follow:
        # https://towardsdatascience.com/using-jupyter-notebook-running-on-a-remote-docker-container-via-ssh-ea2c3ebb9055
        # https://docs.ray.io/en/latest/ray-core/using-ray-with-jupyter.html
        if self.hardware is None:
            raise RuntimeError("Cannot SSH, running locally")

        tunnel, port_fwd = self.hardware.ssh_tunnel(
            local_port=port_forward, num_ports_to_try=10
        )
        try:
            install_cmd = "pip install jupyterlab"
            jupyter_cmd = f"jupyter lab --port {port_fwd} --no-browser"
            # port_fwd = '-L localhost:8888:localhost:8888 '  # TOOD may need when we add docker support
            with self.hardware.pause_autostop():
                self.hardware.run(commands=[install_cmd, jupyter_cmd], stream_logs=True)

        finally:
            if sync_package_on_close:
                if sync_package_on_close == "./":
                    sync_package_on_close = rh_config.rns_client.locate_working_dir()
                pkg = Package.from_string("local:" + sync_package_on_close)
                self.hardware.rsync(
                    source=f"~/{pkg.name}", dest=pkg.local_path, up=False
                )
            if not persist:
                tunnel.stop(force=True)
                kill_jupyter_cmd = f"jupyter notebook stop {port_fwd}"
                self.hardware.run(commands=[kill_jupyter_cmd])

    def keep_warm(
        self,
        autostop_mins=None,
        # TODO regions: List[str] = None,
        # TODO min_replicas: List[int] = None,
        # TODO max_replicas: List[int] = None
    ):
        if autostop_mins is None:
            logger.info(f"Keeping {self.name} indefinitely warm")
            # keep indefinitely warm if user doesn't specify
            autostop_mins = -1
        self.hardware.keep_warm(autostop_mins=autostop_mins)

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
            module_path = Path.cwd() / (f"{name}_fn.py" if name else "send_fn.py")
            logging.info(
                f"Writing out send function to {str(module_path)} as "
                f"functions serialized in notebooks are brittle. Please make "
                f"sure the function does not rely on any local variables, "
                f"including imports (which should be moved inside the function body)."
            )
            if not name:
                logging.warning(
                    "You should name Sends that are created in notebooks to avoid naming collisions "
                    "between the modules that are created to hold their functions "
                    '(i.e. "send_fn.py" errors.'
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
            # fn_pointers = Send.extract_fn_paths(raw_fn=new_fn, reqs=config['reqs'])


def send(
    fn: Optional[Union[str, Callable]] = None,
    name: [Optional[str]] = None,
    hardware: Optional[Union[str, Cluster]] = None,
    reqs: Optional[List[str]] = None,
    setup_cmds: Optional[List[str]] = None,
    # TODO image: Optional[str] = None,
    dryrun: bool = False,
    load_secrets: bool = False,
    serialize_notebook_fn: bool = False,
):
    """Factory constructor to construct the Send for various provider types.

    fn: The function which will execute on the remote cluster when this send is called.
    name: Name of the Send to create or retrieve, either from a local config or from the RNS.
    hardware: Hardware to use for the Send, either a string name of a Cluster object, or a Cluster object.
    package: Package to send to the remote cluster, either a string name of a Package, package url,
        or a Package object.
    reqs: List of requirements to install on the remote cluster, or path to a requirements.txt file. If a list
        of pypi packages is provided, including 'requirements.txt' in the list will install the requirements
        in `package`. By default, if reqs is left as None, we'll set it to ['requirements.txt'], which installs
        just the requirements of package. If an empty list is provided, no requirements will be installed.
    image (TODO): Docker image id to use on the remote cluster, or path to Dockerfile.
    dryrun: Whether to create the Send if it doesn't exist, or load the Send object as a dryrun.
    """

    config = rh_config.rns_client.load_config(name)
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
        fn_pointers = Send.extract_fn_paths(raw_fn=fn, reqs=config["reqs"])
        if fn_pointers[1] == "notebook":
            fn_pointers = Send._handle_nb_fn(
                fn,
                fn_pointers=fn_pointers,
                serialize_notebook_fn=serialize_notebook_fn,
                name=config["name"],
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
        # repo_package = Package(url=f'/',
        #                        fs='github',
        #                        data_config={'org': username, 'repo': repo_name, 'sha': branch_name,
        #                                     'filecache': {'cache_storage': repo_name}},
        #                        install_method='local')
        # config['reqs'] = [repo_package] + config['reqs']

    config["hardware"] = hardware or config.get("hardware")
    if isinstance(config["hardware"], str):
        hw_dict = rh_config.rns_client.load_config(config["hardware"])
        if not hw_dict:
            raise RuntimeError(
                f'Hardware {rh_config.rns_client.resolve_rns_path(config["hardware"])} '
                f"not found locally or in RNS."
            )
        config["hardware"] = hw_dict

    config["setup_cmds"] = (
        setup_cmds if setup_cmds is not None else config.get("setup_cmds")
    )

    config["access_level"] = config.get("access_level", Send.DEFAULT_ACCESS)

    new_send = Send.from_config(config, dryrun=dryrun)

    if load_secrets and not dryrun:
        new_send.send_secrets()

    return new_send
