import asyncio
import copy
import importlib
import inspect
import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor
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
    "set_async",
    "__getattribute__",
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
    "remote",
    "local",
    "resolve",
    "_resolve",
    "resolved_state",
    "fetch",
    "fetch_async",
    "name",
    "rename",
    "rns_address",
    "save",
    "save_attrs_to_config",
    "share",
    "system",
    "to",
    "unname",
    "provenance",
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
        provenance: Optional[dict] = None,
        **kwargs,
    ):
        """
        Runhouse Module object

        .. note::
                To build a Module, please use the factory method :func:`module`.
        """
        super().__init__(name=name, dryrun=dryrun, provenance=provenance, **kwargs)
        self._system = _get_cluster_from(
            system or _current_cluster(key="config"), dryrun=dryrun
        )
        self._env = env
        is_builtin = hasattr(sys.modules["runhouse"], self.__class__.__qualname__)
        if not cls_pointers and not is_builtin:
            # When creating a module as a subclass of rh.Module, we need to collect pointers here
            self._env = env or Env()
            # If we're creating pointers, we're also local to the class definition and package, so it should be
            # set as the workdir (we can do this in a fancier way later)
            self._env.working_dir = self._env.working_dir or "./"
            cls_pointers = Module._extract_pointers(self.__class__, reqs=self._env.reqs)
        self._cls_pointers = cls_pointers
        self._resolve = False

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
        if self._cls_pointers:
            # For some reason sometimes this is coming back as a string, so we force it into a tuple
            config["cls_pointers"] = tuple(self._cls_pointers)
        return config

    @classmethod
    def from_config(cls, config: dict, dryrun=False):
        if config.get("cls_pointers"):
            config.pop("resource_subtype", None)
            logger.debug(f"Constructing module from pointers {config['cls_pointers']}")
            (module_path, module_name, class_name) = config["cls_pointers"]
            module_cls = cls._get_obj_from_pointers(
                module_path, module_name, class_name
            )
            if not issubclass(module_cls, Module):
                # Case when module was created through rh.module(new_class) factory, and needs to be
                # made into a subclass of rh.Module. We'll follow the same flow as the subclass-created module below,
                # where we don't call __init__ explicitly, because __init__ will call the subclass's init and this may
                # a "type" module rather than an "instance". The user might instantiate it later, or it may be
                # populated with attributes by the servlet's put_resource.
                module_cls = _module_subclass_factory(
                    module_cls, config.get("cls_pointers")
                )

            # Module created as subclass of rh.Module may not have rh.Module's
            # constructor signature (e.g. system, env, etc.), so assign them manually
            # We don't call __init__ here because we don't know the signature of the subclass's __init__
            # If this resource was put on a cluster with put_resource, the servlet will be populating the rest
            # of the class-specific attributes.
            new_module = module_cls.__new__(module_cls)
            config = module_cls._check_for_child_configs(config)
            new_module.system = config.pop("system", None)
            new_module.env = config.pop("env", None)
            new_module.name = config.pop("name", None)
            new_module._cls_pointers = config.pop("cls_pointers", None)
            new_module.dryrun = config.pop("dryrun", False)
            new_module.provenance = config.pop("provenance", None)
            return new_module

        if config.get("resource_subtype", None) == "module":
            config = Module._check_for_child_configs(config)
            return Module(**config, dryrun=dryrun)

        # If there are no class pointers, we assume the module is a built-in rh.Module subclass
        resource_class = getattr(
            sys.modules["runhouse"],
            config.pop("resource_subtype").capitalize(),
            None,
        )
        if not resource_class:
            raise TypeError(
                f"Could not find module associated with {config['resource_subtype']}"
            )
        config = resource_class._check_for_child_configs(config)
        return resource_class(**config, dryrun=dryrun)

    @classmethod
    def _check_for_child_configs(cls, config):
        """Overload by child resources to load any resources they hold internally."""
        system = config.get("system")
        if isinstance(system, str):
            config["system"] = _get_cluster_from(system)
        env = config.get("env")
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

    def remote_init(self, *args, **kwargs):
        """A method which you can overload and will be called remotely on the cluster upon initialization there,
        in case you want to do certain initialization activities on the cluster only. For example, if you want
        to load a model or dataset and send it to GPU, you probably don't want to do those locally and send the
        state over to the cluster."""
        if self._cls_pointers:
            (module_path, module_name, class_name) = self._cls_pointers
            module_cls = self._get_obj_from_pointers(
                module_path, module_name, class_name
            )
            module_cls.__init__(self, *args, **kwargs)

    @staticmethod
    def _get_obj_from_pointers(module_path, module_name, obj_name):
        """Helper method to load a class or function from a module path, module name, and class name."""
        if module_path:
            abs_path = str((Path.home() / module_path).expanduser().resolve())
            if abs_path not in sys.path:
                sys.path.append(abs_path)
                logger.debug(f"Appending {module_path} to sys.path")

        if module_name in obj_store.imported_modules:
            importlib.invalidate_caches()
            obj_store.imported_modules[module_name] = importlib.reload(
                obj_store.imported_modules[module_name]
            )
            logger.debug(f"Reloaded module {module_name}")
        else:
            logger.debug(f"Importing module {module_name}")
            obj_store.imported_modules[module_name] = importlib.import_module(
                module_name
            )
        return getattr(obj_store.imported_modules[module_name], obj_name)

    def to(
        self,
        system: Union[str, Cluster],
        env: Optional[Union[str, List[str], Env]] = None,
        name: Optional[str] = None,
    ):
        """Put a copy of the module on the destination system and env, and return the new module.

        Example:
            >>> local_module = rh.module(my_class)
            >>> cluster_module = local_module.to("my_cluster")
            >>> cluster_module = module.to(my_cluster)
        """
        if system == self.system and env == self.env:
            if name and not self.name == name:
                # TODO return duplicate object under new name, don't rename
                self.rename(name)
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
            new_module.name = (
                name
                or self.name
                or (
                    self._cls_pointers[2]
                    if self._cls_pointers
                    else None
                    or _generate_default_name(
                        prefix=self.__class__.__qualname__.lower()
                    )
                )
            )
            if system.on_this_cluster():
                new_module.pin()
            else:
                # We only send over state for instances, not classes
                state = {}
                if not isinstance(self, type):
                    state = {
                        attr: val
                        for attr, val in self.__dict__.items()
                        if attr[0] != "_" and attr not in new_module.config_for_rns
                    }
                system.put_resource(new_module, state, dryrun=True)

        return new_module

    def get_or_to(
        self,
        system: Union[str, Cluster],
        env: Optional[Union[str, List[str], Env]] = None,
        name: Optional[str] = None,
    ):
        name = self.name or name
        if not name:
            raise ValueError(
                "You must specify a name for the module if you want to get_or_to it."
            )
        system = _get_cluster_from(system)
        try:
            remote = system.get(name, remote=True)
        except KeyError:
            remote = None
        if remote:
            return remote
        self.name = name
        return self.to(system, env)

    def __getattribute__(self, item):
        """Override to allow for remote execution if system is a remote cluster. If not, the subclass's own
        __getattr__ will be called."""
        if item in LOCAL_METHODS or not hasattr(self, "_system"):
            return super().__getattribute__(item)
        system = super().__getattribute__("_system")

        attr = super().__getattribute__(item)

        if not system or not isinstance(system, Cluster) or system.on_this_cluster():
            return attr

        # Don't try to run private methods or attributes remotely
        if item[0] == "_":
            return attr

        name = super().__getattribute__("_name")

        # Handle properties
        if not callable(attr):
            # TODO should we throw a warning here or is that annoying?
            return attr

        signature = inspect.signature(attr)
        has_local_arg = "local" in signature.parameters
        local_default_true = (
            has_local_arg and signature.parameters["local"].default is True
        )

        # Needed to handle async Functions, because we can't detect if the function they wrap is async like we
        # do below for Module methods
        try:
            is_async = (
                super().__getattribute__("_is_async") if item == "call" else False
            )
            is_async_gen = (
                super().__getattribute__("_is_async_gen") if item == "call" else False
            )
        except AttributeError:
            is_async = False
            is_async_gen = False

        class RemoteMethodWrapper:
            """Helper class to allow methods to be called with __call__, remote, or run."""

            def __call__(self, *args, **kwargs):
                # stream_logs and run_name are both supported args here, but we can't include them explicitly because
                # the local code path here will throw an error if they are included and not supported in the
                # method signature.

                # Check if the method has a "local=True" arg, and check that the user didn't pass local=False instead
                if local_default_true and kwargs.pop("local", True):
                    return attr(*args, **kwargs)

                # If the method is a coroutine, we need to wrap it in a function so we can await it
                if (
                    inspect.iscoroutinefunction(attr)
                    or inspect.isasyncgenfunction(attr)
                    or is_async
                ):

                    def call_wrapper():
                        return system.call(
                            name,
                            item,
                            *args,
                            **kwargs,
                        )

                    if inspect.isasyncgenfunction(attr) or is_async_gen:

                        async def async_gen():
                            for res in call_wrapper():
                                yield res

                        return async_gen()

                    _executor = ThreadPoolExecutor(1)

                    async def async_call():
                        return await loop.run_in_executor(_executor, call_wrapper)

                    loop = asyncio.get_event_loop()
                    return asyncio.create_task(async_call())

                return system.call(
                    name,
                    item,
                    *args,
                    **kwargs,
                )

            def remote(self, *args, stream_logs=True, run_name=None, **kwargs):
                return self.__call__(
                    *args,
                    stream_logs=stream_logs,
                    run_name=run_name,
                    remote=True,
                    **kwargs,
                )

            def run(self, *args, stream_logs=False, run_name=None, **kwargs):
                return self.__call__(
                    *args,
                    stream_logs=stream_logs,
                    run_name=run_name,
                    run_async=True,
                    **kwargs,
                )

            def local(self, *args, **kwargs):
                """Allows us to call a function with fn.local(*args) instead of fn(*args, local=True)"""
                return self.__call__(
                    *args,
                    local=True,
                    **kwargs,
                )

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
            or not self._name
        ):
            return super().__setattr__(key, value)

        return self._system.call(
            module_name=self._name,
            method_name=key,
            new_value=value,
            stream_logs=False,
        )

    def refresh(self):
        """Update the resource in the object store."""
        if not self.system or not self.name:
            return self
        if self._system.on_this_cluster():
            return obj_store.get(self._name)
        elif isinstance(self._system, Cluster):
            return self._system.get(self._name, remote=True)
        else:
            return self

    @property
    def remote(self):
        """Helper property to allow for access to remote properties, both public and private. Returning functions
        is not advised.

        Example:
            >>> my_module.remote.my_property
            >>> my_module.remote._my_private_property
            >>> my_module.remote.size = 14
        """
        system = super().__getattribute__("_system")
        name = super().__getattribute__("_name")

        outer_super_gettattr = super().__getattribute__
        outer_super_setattr = super().__setattr__

        class RemotePropertyWrapper:
            @classmethod
            def __getattribute__(cls, item):
                if not isinstance(system, Cluster) or not name:
                    return outer_super_gettattr(item)

                if isinstance(system, Cluster) and name and system.on_this_cluster():
                    obj_store_obj = obj_store.get(name, check_other_envs=True)
                    if obj_store_obj:
                        return obj_store_obj.__getattribute__(item)
                    else:
                        return self.__getattribute__(item)

                return system.call(name, item, stream_logs=False)

            @classmethod
            def __setattr__(cls, key, value):
                if (
                    not isinstance(system, Cluster)
                    or not name
                    or system.on_this_cluster()
                ):
                    return outer_super_setattr(key, value)

                return system.call(
                    module_name=name,
                    method_name=key,
                    new_value=value,
                    stream_logs=False,
                )

            @classmethod
            def __call__(cls, *args, **kwargs):
                return system.get(name, *args, **kwargs)

        return RemotePropertyWrapper()

    @property
    def local(self):
        """Helper property to allow for access to local properties, both public and private.

        Example:
            >>> my_module.local.my_property
            >>> my_module.local._my_private_property

            >>> my_module.local.size = 14
        """
        outer_super_gettattr = super().__getattribute__
        outer_super_setattr = super().__setattr__

        class LocalPropertyWrapper:
            @classmethod
            def __getattribute__(cls, item):
                return outer_super_gettattr(item)

            @classmethod
            def __setattr__(cls, key, value):
                return outer_super_setattr(key, value)

        return LocalPropertyWrapper()

    def fetch(self, item: str = None, stream_logs: bool = False):
        """Helper method to allow for access to remote state, both public and private. Fetching functions
        is not advised. `system.get(module.name).resolved_state()` is roughly equivalent to `module.fetch()`.

        Example:
            >>> my_module.fetch("my_property")
            >>> my_module.fetch("my_private_property")

            >>> MyRemoteClass = rh.module(my_class).to(system)
            >>> MyRemoteClass(*args).fetch() # Returns a my_class instance, populated with the remote state

            >>> my_blob.fetch() # Returns the data of the blob, due to overloaded ``resolved_state`` method

            >>> class MyModule(rh.Module):
                    # ...
            >>> MyModule(*args).to(system).fetch() # Returns the full remote module, including private and public state
        """
        system = super().__getattribute__("_system")
        name = super().__getattribute__("_name")

        if item is not None:
            if not isinstance(system, Cluster) or not name:
                return super().__getattribute__(item)

            if isinstance(system, Cluster) and name and system.on_this_cluster():
                try:
                    obj_store_obj = obj_store.get(
                        name, check_other_envs=True, default=KeyError
                    )
                    return obj_store_obj.__getattribute__(item)
                except KeyError:
                    return self.__getattribute__(item)

            return system.call(name, item, stream_logs=stream_logs)
        else:
            if not isinstance(system, Cluster) or not name:
                return self.resolved_state()
            return system.get(name, stream_logs=stream_logs).resolved_state()

    async def fetch_async(
        self, key: str, remote: bool = False, stream_logs: bool = False
    ):
        """Async version of fetch. Can't be a property like `fetch` because __getattr__ can't be awaited.

        Example:
            >>> await my_module.fetch_async("my_property")
            >>> await my_module.fetch_async("_my_private_property")
        """
        system = super().__getattribute__("_system")
        name = super().__getattribute__("_name")

        def call_wrapper():
            if not key:
                return system.get(name, remote=remote, stream_logs=stream_logs)

            if isinstance(system, Cluster) and name and system.on_this_cluster():
                obj_store_obj = obj_store.get(name, check_other_envs=True)
                if obj_store_obj:
                    return obj_store_obj.__getattribute__(key)
                else:
                    return self.__getattribute__(key)
            return system.call(name, key, remote=remote, stream_logs=stream_logs)

        if (
            key
            and hasattr(self, key)
            and inspect.isasyncgenfunction(super().__getattribute__(key))
        ):

            async def async_gen():
                for res in call_wrapper():
                    yield res

            return async_gen()

        _executor = ThreadPoolExecutor(1)

        async def async_call():
            return await loop.run_in_executor(_executor, call_wrapper)

        loop = asyncio.get_event_loop()
        return await asyncio.create_task(async_call())

    async def set_async(self, key: str, value):
        """Async version of property setter.

        Example:
            >>> await my_module.set_async("my_property", my_value)
            >>> await my_module.set_async("_my_private_property", my_value)
        """
        if (
            not self._system
            or not isinstance(self._system, Cluster)
            or self._system.on_this_cluster()
            or not self._name
        ):
            return super().__setattr__(key, value)

        def call_wrapper():
            return self._system.call(
                module_name=self._name,
                method_name=key,
                new_value=value,
                stream_logs=False,
            )

        _executor = ThreadPoolExecutor(1)

        async def async_call():
            return await loop.run_in_executor(_executor, call_wrapper)

        loop = asyncio.get_event_loop()
        return await asyncio.create_task(async_call())

    def resolve(self):
        """Specify that the module should resolve to a particular state when passed into a remote method. This is
        useful if you want to revert the module's state to some "Runhouse-free" state once it is passed into a
        Runhouse-unaware function. For example, if you call a Runhouse-unaware function with `.remote()`,
        you will be returned a Blob which wraps your data. If you want to pass that Blob into another function
        that operates on the original data (e.g. a function that takes a numpy array), you can call
        `my_second_fn(my_blob.resolve())`, and `my_blob` will be replaced with the contents of its `.data` on the
        cluster before being passed into `my_second_fn`.

        Resolved state is defined by the `resolved_state` method. By default, modules created with the
        `module` factory constructor will be resolved to their original non-module-wrapped class (or best attempt).
        Modules which are defined as a subclass of `Module` will be returned as-is, as they have no other
        "original class."

        Example:
            >>> my_module = rh.module(my_class)
            >>> my_remote_fn(my_module.resolve()) # my_module will be replaced with the original class `my_class`

            >>> my_result_blob = my_remote_fn.remote(args)
            >>> my_other_remote_fn(my_result_blob.resolve()) # my_result_blob will be replaced with its data

        """
        self._resolve = True
        return self

    def resolved_state(self):
        """Return the resolved state of the module. By default, this is the original class of the module if it was
        created with the `module` factory constructor."""
        if not self._cls_pointers:
            self._resolve = False
            return self

        (module_path, module_name, class_name) = self._cls_pointers
        original_class = self._get_obj_from_pointers(
            module_path, module_name, class_name
        )
        if issubclass(original_class, Module):
            self._resolve = False
            return self

        if not self.__dict__:
            # This is a non-instantiated Module, i.e. represents a class rather than an instance
            return original_class

        new_module = original_class.__new__(original_class)
        # TODO pop out any attributes that are not in the original class?
        new_module.__dict__ = self.__dict__
        return new_module

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
        if not self.system or not isinstance(self.system, Cluster):
            return
        if self.system.on_this_cluster():
            obj_store.rename(old_key=old_name, new_key=self.name)
        else:
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
                self.name = name
                if isinstance(self.system, Cluster):
                    self.system.put_resource(self)
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


def _module_subclass_factory(cls, pointers, signature=None):
    def __init__(
        self,
        *args,
        system=None,
        env=None,
        dryrun=False,
        cls_pointers=pointers,
        name=None,
        provenance=None,
        **kwargs,
    ):
        # args and kwargs are passed to the cls's __init__ method if this is being called on a cluster. They
        # shouldn't be passed otherwise.
        Module.__init__(
            self,
            cls_pointers=cls_pointers,
            name=name,
            system=system,
            env=env,
            dryrun=dryrun,
            provenance=provenance,
        )
        # This allows a class which is already on the cluster to construct an instance of itself with a factory
        # method, e.g. my_module = MyModuleCls.factory_constructor(*args, **kwargs)
        if self.system and self.system.on_this_cluster():
            self.remote_init(*args, **kwargs)

    def __call__(
        self,
        *args,
        dryrun=False,
        name=None,
        **kwargs,
    ):
        # TODO change setting logic to be "mod.local.x = 5" or "mod.remote.x = 5", with properties being remote by
        # default and private methods being local by default for both setting and getting
        new_module = copy.copy(self)
        # Create a copy of the item on the cluster under the new name
        new_module.name = name or self.name
        new_module.dryrun = dryrun
        if not new_module.dryrun:
            new_module.system.put_resource(new_module)
            new_module.remote_init(*args, **kwargs)

        return new_module

    methods = {"__init__": __init__, "__call__": __call__}
    new_type = type(pointers[2], (Module, cls), methods)
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
            e.g. ``key = model.forward.run(x)`` (which returns a key to retrieve the result with
            ``cluster.get(key)``), or with ``run_obj = model.train.remote(x)``, which runs synchronously but returns
            a remote object to avoid passing heavy results back over the network.
        - Setting attributes, both public and private, will be executed remotely, with the new values only being
            set in the remote module and not the local one. This excludes any methods or attribtes of the Module class
            proper (e.g. ``system`` or ``name``), which will be set locally.
        - Attributes, private properties can be fetched with the ``remote`` property, and the full resource can be
            fetched using ``.fetch()``, e.g. ``model.remote.weights``, ``model.remote.__dict__``, ``model.fetch()``.
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
        >>> import transformers
        >>> import json
        >>>
        >>> class Model(rh.Module):
        >>>    def __init__(self, model_id, device="cpu", system=None, env=None):
        >>>        super().__init__(system=system, env=env)
        >>>        self.model_id = model_id
        >>>        self.device = device
        >>>
        >>>    def remote_init(self):
        >>>        self.tokenizer = transformers.AutoTokenizer.from_pretrained(self.model_id)
        >>>        self.model = transformers.AutoModel.from_pretrained(self.model_id).to(self.device)
        >>>
        >>>    def predict(self, x):
        >>>        x = self.tokenizer(x, return_tensors="pt")
        >>>        return self.model(x)
        >>>
        >>> model = Model(model_id="bert-base-uncased", device="cuda", system="my_gpu", env="my_env")
        >>> model.predict("Hello world!")   # Runs on system in env
        >>> tok = model.remote.tokenizer     # Returns remote tokenizer
        >>> model_id = model.model_id       # Returns local model_id (not remote)
        >>> model.fetch()                   # Returns full remote module, including model and tokenizer
        >>>
        >>> # You can also create a model locally and then send it to a cluster with .to
        >>> # remote_init will not be called until the model lands on a cluster
        >>> other_model = Model(model_id="bert-base-uncased", device="cuda").to("my_gpu", "my_env")
        >>>
        >>> # Another method: Create a module from an existing class which is not a subclass of Module
        >>> RemoteModel = rh.module(cls=BERTModel, model_id="remote_model", system="my_gpu", env="my_env")
        >>> remote_model = RemoteModel(model_id="bert-base-uncased", device="cuda")
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
    name = name or (
        pointers[2] if pointers else _generate_default_name(prefix="module")
    )
    signature = inspect.signature(cls)

    # return _module_subclass_factory(cls, pointers, system, env, name, signature)
    module_subclass = _module_subclass_factory(cls, pointers, signature)
    return module_subclass(
        system=system, env=env, dryrun=dryrun, cls_pointers=pointers, name=name
    )
