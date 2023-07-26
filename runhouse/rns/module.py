import copy
import functools
import importlib
import inspect
import logging
import os
import sys
from pathlib import Path
from typing import Callable, List, Optional, Tuple, Type, Union

from runhouse.rh_config import obj_store, rns_client
from runhouse.rns.envs import Env
from runhouse.rns.hardware.cluster import Cluster
from runhouse.rns.packages import Package
from runhouse.rns.resource import Resource
from runhouse.rns.utils.env import _get_env_from
from runhouse.rns.utils.hardware import _current_cluster, _get_cluster_from
from runhouse.rns.utils.names import _generate_default_name

logger = logging.getLogger(__name__)

# These are methods that the Module's __getattribute__ logic should not intercept to run remotely
LOCAL_METHODS = [
    "RESOURCE_TYPE",
    "__class__",
    "__delattr__",
    "__dict__",
    "__dir__",
    "__doc__",
    "__eq__",
    "__format__",
    "__ge__",
    "__gt__",
    "__hash__",
    "__init__",
    "__init_subclass__",
    "__le__",
    "__lt__",
    "__module__",
    "__ne__",
    "__new__",
    "__reduce__",
    "__reduce_ex__",
    "__repr__",
    "__setattr__",
    "__sizeof__",
    "__str__",
    "__subclasshook__",
    "__weakref__",
    "_check_for_child_configs",
    "_cls_pointers",
    "_env",
    "_extract_pointers",
    "_name",
    "_resource_string_for_subconfig",
    "_rns_folder",
    "_save_sub_resources",
    "_system",
    "config_for_rns",
    "delete_configs",
    "dryrun",
    "env",
    "from_config",
    "from_name",
    "history",
    "is_local",
    "fetch",
    "name",
    "rename",
    "rns_address",
    "save",
    "save_attrs_to_config",
    "share",
    "system",
    "to",
    "unname",
]


class Module(Resource):
    RESOURCE_TYPE = "module"

    def __init__(
            self,
            cls_pointers: Optional[Tuple] = None,
            name: Optional[str] = None,
            system: Union[Cluster] = None,
            env: Optional[Env] = None,
            dryrun: bool = False,
            **kwargs,
    ):
        """
        Runhouse Module object

        .. note::
                To build a Module, please use the factory method :func:`module`.
        """
        super().__init__(name=name, dryrun=dryrun)
        self._system = system
        self._env = env
        if not cls_pointers:
            # When creating a module as a subclass of rh.Module, we need to collect pointers here
            self._env = env or Env()
            # If we're creating pointers, we're also local to the class definition and package, so it should be
            # set as the workdir (we can do this in a fancier way later)
            self._env.working_dir = self._env.working_dir or "./"
            cls_pointers = Module._extract_pointers(self.__class__, reqs=self._env.reqs)
        self._cls_pointers = cls_pointers

    @property
    def config_for_rns(self):
        if not self.system:
            raise ValueError(
                "Cannot save an in-memory local module to RNS. Please send the module to a local "
                "path or system first."
            )
        config = super().config_for_rns
        config["system"] = (
            self._resource_string_for_subconfig(self.system) if self.system else None
        )
        config["env"] = (
            self._resource_string_for_subconfig(self.env) if self.env else None
        )
        config["cls_pointers"] = self._cls_pointers
        return config

    @staticmethod
    def from_config(config: dict, dryrun=False):
        config.pop("resource_type", None)
        config.pop("provenance", None)
        config.pop("resource_subtype", None)

        if config.get("cls_pointers"):
            logger.info(f"Constructing module from pointers {config['cls_pointers']}")
            (module_path, module_name, class_name) = config["cls_pointers"]
            if module_path:
                sys.path.append(module_path)
                logger.info(f"Appending {module_path} to sys.path")

            if module_name in obj_store.imported_modules:
                importlib.invalidate_caches()
                obj_store.imported_modules[module_name] = importlib.reload(
                    obj_store.imported_modules[module_name]
                )
                logger.info(f"Reloaded module {module_name}")
            else:
                logger.info(f"Importing module {module_name}")
                obj_store.imported_modules[module_name] = importlib.import_module(
                    module_name
                )
            cls = getattr(obj_store.imported_modules[module_name], class_name)
            if not issubclass(cls, Module):
                # Case when module was created through rh.module(new_class) factory, and needs to be
                # made into a subclass of rh.Module
                module_cls = _module_subclass_factory(cls, config.get("cls_pointers"))
            else:
                # Case when module was created through NewClass(rh.Module) and is already a Module subclass
                module_cls = cls

            if hasattr(module_cls, "_check_for_child_configs"):
                config = module_cls._check_for_child_configs(config)

            if not issubclass(cls, Module):
                # When we explicitly make module_cls a Molue subclass, we also made sure it could accept all of
                # rh.Module's constructor arguments.
                new_module = module_cls(**config, dryrun=dryrun)
            else:
                # Module created as subclass of rh.Module may not have rh.Module's
                # constructor signature (e.g. system, env, etc.), so assign them manually
                new_module = module_cls()
                new_module.system = config.pop("system", None)
                new_module.env = config.pop("env", None)
                new_module.name = config.pop("name", None)
                new_module._cls_pointers = config.pop("cls_pointers", None)
                new_module.dryrun = config.pop("dryrun", False)
        else:
            config = Module._check_for_child_configs(config)
            new_module = Module(**config, dryrun=dryrun)

        return new_module

    @classmethod
    def _check_for_child_configs(cls, config):
        """Overload by child resources to load any resources they hold internally."""
        system = config["system"]
        if isinstance(system, str):
            config["system"] = _get_cluster_from(system)
        env = config["env"]
        if isinstance(env, str):
            config["env"] = _get_env_from(env)
        return config

    @property
    def system(self):
        return self._system

    @system.setter
    def system(self, new_system: Union[str, Cluster]):
        self._system = _get_cluster_from(new_system)

    @property
    def env(self):
        return self._env

    @env.setter
    def env(self, new_env: Optional[Union[str, Env]]):
        self._env = _get_env_from(new_env)

    def remote_init(self):
        """A method which you can overload and will be called remotely on the cluster upon initialization there,
        in case you want to do certain initialization activities on the cluster only. For example, if you want
        to load a model or dataset and send it to GPU, you probably don't want to do those locally and send the
        state over to the cluster."""
        pass

    def to(
            self,
            system: Union[str, Cluster],
            env: Optional[Union[str, List[str], Env]] = None,
    ):
        """Put a copy of the module on the destination system and env, and return the new module.

        Example:
            >>> local_module = rh.module(my_class)
            >>> cluster_module = local_module.to("my_cluster")
            >>> cluster_module = module.to(my_cluster)
        """
        if system == self.system and env == self.env:
            return self

        if system == "here":
            current_cluster_config = _current_cluster(key="config")
            if current_cluster_config:
                system = Cluster.from_config(current_cluster_config)
            else:
                system = None

        system = (
            _get_cluster_from(system, dryrun=self.dryrun) if system else self.system
        )
        env = env or self.env
        env = _get_env_from(env)

        if env and system:
            env = env.to(system)

        # We need to backup the system here so the __getstate__ method of the cluster
        # doesn't wipe the client of this function's cluster when deepcopy copies it.
        hw_backup = self.system
        self.system = None
        new_module = copy.deepcopy(self)
        self.system = hw_backup

        new_module.system = system
        new_module.env = env
        new_module.dryrun = True

        if isinstance(system, Cluster):
            state = {
                attr: val
                for attr, val in self.__dict__.items()
                if attr[0] != "_" and attr not in new_module.config_for_rns
            }
            new_module.name = (
                self.name or self._cls_pointers[2]
                if self._cls_pointers
                else None
                     or _generate_default_name(prefix=self.__class__.__qualname__.lower())
            )
            system.put_resource(new_module, state, dryrun=True)

        return new_module

    def __getattribute__(self, item):
        """Override to allow for remote execution if system is a remote cluster. If not, the subclass's own
        __getattr__ will be called."""
        if item in LOCAL_METHODS or not hasattr(self, "_system"):
            return super().__getattribute__(item)
        system = super().__getattribute__("_system")
        if not system or not isinstance(system, Cluster) or system.on_this_cluster():
            return super().__getattribute__(item)

        attr = super().__getattribute__(item)

        # Don't try to run private methods or attributes remotely
        if item[0] == "_":
            return attr

        name = super().__getattribute__("_name")

        # Handle properties
        if not callable(attr):
            # TODO should we throw a warning here or is that annoying?
            return attr
            # return system.call_module_method(name, item)

        signature = inspect.signature(attr)
        has_local_arg = "local" in signature.parameters
        local_default_true = (
                has_local_arg and signature.parameters["local"].default is True
        )

        class RemoteMethodWrapper:
            """Helper class to allow methods to be called with __call__, remote, or run."""

            def __call__(self, *args, stream_logs=True, run_name=None, **kwargs):
                # Check if the method has a "local=True" arg, and check that the user didn't pass local=False instead
                if local_default_true and kwargs.pop("local", True):
                    return attr(*args, **kwargs)

                return system.call_module_method(
                    name,
                    item,
                    stream_logs=stream_logs,
                    run_name=run_name,
                    *args,
                    **kwargs,
                )

            def remote(self, *args, stream_logs=True, run_name=None, **kwargs):
                # TODO add a callback? e.g. model.forward.remote(x, callback=print)
                return self.__call__(
                    *args,
                    stream_logs=stream_logs,
                    run_name=run_name,
                    remote=True,
                    **kwargs,
                )

            def run(self, *args, stream_logs=True, run_name=None, **kwargs):
                key = self.remote(
                    *args, stream_logs=stream_logs, run_name=run_name, **kwargs
                )
                return system.get_run(key)

        return RemoteMethodWrapper()

    def __setattr__(self, key, value):
        """Override to allow for remote execution if system is a remote cluster. If not, the subclass's own
        __setattr__ will be called."""
        if key in LOCAL_METHODS or not hasattr(self, "_system"):
            return super().__setattr__(key, value)
        if (
                not self._system
                or not isinstance(self._system, Cluster)
                or self._system.on_this_cluster()
        ):
            return super().__setattr__(key, value)

        return self.system.call_module_method(
            module_name=self.name,
            method_name=key,
            new_value=value,
        )

    @property
    def fetch(self):
        """Helper property to allow for access to remote properties, both public and private. Returning functions
        is not advised.

        Example:
            >>> my_module.fetch.my_property
            >>> my_module.fetch._my_private_property
        """
        # TODO is remote a better name?
        system = super().__getattribute__("_system")
        name = super().__getattribute__("_name")

        class RemotePropertyWrapper:
            @classmethod
            def __getattribute__(cls, item):
                return system.call_module_method(name, item)

            @classmethod
            def __call__(cls, *args, **kwargs):
                return system.get(name, *args, **kwargs)

        return RemotePropertyWrapper()

    # TODO do we need this?
    # def __delattr__(self, key):
    #     if key in LOCAL_METHODS or not hasattr(self, "_system"):
    #         return super().__delattr__(key)
    #     if (
    #         not self._system
    #         or not isinstance(self._system, Cluster)
    #         or self._system.on_this_cluster()
    #     ):
    #         return super().__delattr__(key)
    #
    #     return self.system.call_module_method(
    #         module_name=self.name,
    #         method_name=key,
    #         delete=True,
    #     )

    def _save_sub_resources(self):
        if isinstance(self.system, Resource):
            self.system.save()
        if isinstance(self.env, Resource):
            self.env.save()

    def rename(self, name: str):
        """Rename the module."""
        if self.name == name:
            return
        old_name = self.name
        self.name = name  # Goes through Resource setter to parse name properly (e.g. if rns path)
        if self.system.on_this_cluster():
            obj_store.rename(old_key=old_name, new_key=self.name)
        elif isinstance(self.system, Cluster):
            self.system.rename(old_key=old_name, new_key=self.name)

    def save(
            self,
            name: str = None,
            overwrite: bool = True,
    ):
        # Need to override Resource's save to handle key changes in the obj store
        # Also check that this is a Blob and not a File
        if name and not self.name == name:
            if overwrite:
                self.rename(name)
            else:
                if isinstance(self.system, Cluster):
                    self.system.put(name, self)
        return super().save(name=name, overwrite=overwrite)

    @staticmethod
    def _extract_pointers(raw_cls_or_fn: Union[Type, Callable], reqs: List[str]):
        """Get the path to the module, module name, and function name to be able to import it on the server"""
        if not (isinstance(raw_cls_or_fn, type) or isinstance(raw_cls_or_fn, Callable)):
            raise TypeError(
                f"Expected Type or Callable but received {type(raw_cls_or_fn)}"
            )
        # Background on all these dunders: https://docs.python.org/3/reference/import.html
        py_module = inspect.getmodule(raw_cls_or_fn)

        # Need to resolve in case just filename is given
        module_path = (
            str(Path(inspect.getfile(py_module)).resolve())
            if hasattr(py_module, "__file__")
            else None
        )

        # TODO better way of detecting if in a notebook or interactive Python env
        if not module_path or module_path.endswith("ipynb"):
            # The only time __file__ wouldn't be present is if the function is defined in an interactive
            # interpreter or a notebook. We can't import on the server in that case, so we need to cloudpickle
            # the fn to send it over. The __call__ function will serialize the function if we return it this way.
            # This is a short-term hack.
            # return None, "notebook", raw_fn.__name__
            root_path = os.getcwd()
            module_name = "notebook"
            cls_or_fn_name = raw_cls_or_fn.__name__
        else:
            root_path = os.path.dirname(module_path)
            module_name = inspect.getmodulename(module_path)
            # TODO __qualname__ doesn't work when fn is aliased funnily, like torch.sum
            cls_or_fn_name = getattr(
                raw_cls_or_fn, "__qualname__", raw_cls_or_fn.__name__
            )

            # Adapted from https://github.com/modal-labs/modal-client/blob/main/modal/_function_utils.py#L94
            if getattr(py_module, "__package__", None):
                module_path = os.path.abspath(py_module.__file__)
                package_paths = [
                    os.path.abspath(p)
                    for p in __import__(py_module.__package__).__path__
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
                module_name = py_module.__spec__.name

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
                        else Path(rns_client.locate_working_dir()) / req
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

        return remote_import_path, module_name, cls_or_fn_name

    # Found in python decorator logic, maybe use
    # func_name = getattr(f, '__qualname__', f.__name__)
    # module_name = getattr(f, '__module__', '')
    # if module_name:
    #     full_name = f'{module_name}.{func_name}'
    # else:
    #     full_name = func_name


def _module_subclass_factory(
        cls, pointers, new_system=None, new_env=None, resource_name=None, signature=None
):
    def __init__(
            self,
            system=None,
            env=None,
            dryrun=False,
            cls_pointers=None,
            name=resource_name,
            **kwargs,
    ):
        cls.__init__(self, **kwargs)
        Module.__init__(
            self, cls_pointers=cls_pointers or pointers, name=name, dryrun=dryrun
        )
        # Overwrite the .to classmethod we injected below, we don't need it anymore
        self.to = functools.partial(Module.to, self)
        if not self.dryrun:
            self.to(system, env)
        self.system = system
        self.env = env

    @classmethod
    def to(new_cls, system=new_system, env=new_env):
        new_cls.__init__ = functools.partialmethod(__init__, system=system, env=env)
        return new_cls

    methods = {
        k: v for k, v in cls.__dict__.items() if callable(v) and not k.startswith("__")
    }
    methods.update({"__init__": __init__})
    new_type = type(pointers[2], (Module, cls), methods)
    # Classmethod, so you can do RemoteClass = rh.module(MyClass).to("my_cluster", "prod")
    new_type.to = to
    new_type.__signature__ = signature
    return new_type


def module(
        cls: [Type] = None,
        name: Optional[str] = None,
        system: Optional[Union[str, Cluster]] = None,
        env: Optional[Union[str, Env]] = None,
        dryrun: bool = False,
):
    """Returns a Module object, which can be used to instantiate and interact with the class remotely.

    The behavior of Modules (and subclasses thereof) is as follows:
        - Any callable public method of the module is intercepted and executed remotely over rpc, with exception of
            certain functions Python doesn't make interceptable (e.g. __call__, __init__), and methods of the Module
            class (e.g. ``to``, ``fetch``, etc.). Properties and private methods are not intercepted, and will be
            executed locally.
        - Any method which executes remotely may be called normally, e.g. ``model.forward(x)``, or asynchronously,
            e.g. ``key = model.forward.remote(x)`` (which returns a key to retrieve the result with
            ``cluster.get(key)``), or with ``run_obj = model.train.run(x)``, which is also async but returns a
            Run object instead of a key.
        - Setting attributes, both public and private, will be executed remotely, with the new values only being
            set in the remote module and not the local one. This excludes any methods or attribtes of the Module class
            proper (e.g. ``system`` or ``name``), which will be set locally.
        - Attributes, private properties, and the full resource can be fetched over rpc using the ``.fetch`` property,
            e.g. ``model.fetch.weights``, ``model.fetch.__dict__``, ``model.fetch()``.
        - When a module is sent to a cluster, it's public attribtes are serialized, sent over, and repopulated in the
            remote instance. This means that any changes to the module's attributes will not be reflected in the remote



    Args:
        cls: The class to instantiate.
        name (Optional[str]): Name to give the module object, to be reused later on.
        system (Optional[str or Cluster]): File system or cluster name. If providing a file system this must be one of:
            [``file``, ``github``, ``sftp``, ``ssh``, ``s3``, ``gs``, ``azure``].
            We are working to add additional file system support. If providing a cluster, this must be a cluster object
            or name, and whether the data is saved to the object store or filesystem depends on whether a path is
            specified.
        env (Optional[str or Env]): Environment in which the module should live on the cluster, if system is cluster.
        dryrun (bool): Whether to create the Blob if it doesn't exist, or load a Blob object as a dryrun.
            (Default: ``False``)

    Returns:
        Module: The resulting module.

    Example:
        >>> import runhouse as rh
        >>> import json
        >>>
        >>> class Model(rh.Module):
        >>>    def __init__(self, name, device="cpu", system=None, env=None):
        >>>        super().__init__(system=system, env=env)
        >>>        self.tokenizer = transformers.AutoTokenizer.from_pretrained(name)
        >>>        self.model = transformers.AutoModel.from_pretrained(name).to(device)
        >>>
        >>>    def predict(self, x):
        >>>        x = self.tokenizer(x, return_tensors="pt")
        >>>        return self.model(x)
        >>>
        >>> model = Model(name="bert-base-uncased", device="cuda", system="my_gpu", env="my_env")
        >>> model.predict("Hello world!")  # Runs on system in env
        >>>
        >>> # Create a module from a class
        >>> RemoteModel = rh.module(cls=Model, name="remote_model", system="my_gpu", env="my_env")
        >>> remote_model = RemoteModel(name="bert-base-uncased", device="cuda")
        >>> remote_model.predict("Hello world!")  # Runs on system in env
        >>>
        >>> # Loading a module
        >>> my_local_module = rh.module(name="~/my_module")
        >>> my_s3_module = rh.module(name="@/my_module")
    """
    if name and not any([cls, system, env]):
        # Try reloading existing module
        return Module.from_name(name, dryrun)

    system = _get_cluster_from(system or _current_cluster(key="config"), dryrun=dryrun)

    if not isinstance(env, Env):
        env = _get_env_from(env) or Env()
        env.working_dir = env.working_dir or "./"

    pointers = Module._extract_pointers(cls, env.reqs)
    name = name or _generate_default_name(prefix=pointers[2] if pointers else "module")
    signature = inspect.signature(cls)

    return _module_subclass_factory(cls, pointers, system, env, name, signature)
