import asyncio
import copy
import importlib
import inspect
import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Type, Union

from apispec import APISpec
from pydantic import create_model

from runhouse.globals import obj_store, rns_client
from runhouse.resources.envs import _get_env_from, Env
from runhouse.resources.hardware import _current_cluster, _get_cluster_from, Cluster
from runhouse.resources.packages import Package
from runhouse.resources.resource import Resource
from runhouse.rns.utils.api import ResourceAccess, ResourceVisibility
from runhouse.rns.utils.names import _generate_default_name
from runhouse.servers.http import HTTPClient
from runhouse.servers.http.http_utils import CallParams

logger = logging.getLogger(__name__)

# These are attributes that the Module's __getattribute__ logic should not intercept to run remotely
# and values that shouldn't be passed as state in put_resource
MODULE_ATTRS = [
    "_pointers",
    "_endpoint",
    "_client",
    "_visibility",
    "_env",
    "_name",
    "_rns_folder",
    "_system",
    "dryrun",
    "_resolve",
    "provenance",
    "_signature",
    "_dumb_signature_cache",
]


class Module(Resource):
    RESOURCE_TYPE = "module"

    def __init__(
        self,
        pointers: Optional[Tuple] = None,
        signature: Optional[dict] = None,
        endpoint: Optional[str] = None,
        name: Optional[str] = None,
        system: Union[Cluster, str] = None,
        env: Optional[Env] = None,
        dryrun: bool = False,
        provenance: Optional[dict] = None,
        **kwargs,
    ):
        """
        Runhouse Module object
        """
        super().__init__(name=name, dryrun=dryrun, provenance=provenance, **kwargs)
        self._system = _get_cluster_from(
            system or _current_cluster(key="config"), dryrun=dryrun
        )
        self._env = env
        is_builtin = hasattr(sys.modules["runhouse"], self.__class__.__qualname__)
        if not pointers and not is_builtin:
            # If there are no pointers and this isn't a builtin module, we assume this is a user-created subclass
            # of rh.Module, and we need to do the factory constructor logic here.

            # When creating a module as a subclass of rh.Module, we need to collect pointers here
            self._env = self._env or Env(name=Env.DEFAULT_NAME)
            # If we're creating pointers, we're also local to the class definition and package, so it should be
            # set as the workdir (we can do this in a fancier way later)
            self._env.working_dir = self._env.working_dir or "./"
            pointers = Module._extract_pointers(self.__class__, reqs=self._env.reqs)
        self._pointers = pointers
        self._endpoint = endpoint
        self._signature = signature
        self._dumb_signature_cache = None
        self._resolve = False
        self._openapi_spec = None

    def config(self, condensed=True):
        if not self.system:
            raise ValueError(
                "Cannot save an in-memory local module to RNS. Please send the module to a local "
                "path or system first."
            )
        config = super().config(condensed)
        if self.system:
            system = self._resource_string_for_subconfig(self.system, condensed)
        else:
            system = None

        config["system"] = system
        config["env"] = (
            self._resource_string_for_subconfig(self.env, condensed)
            if self.env
            else None
        )
        if self._pointers:
            # For some reason sometimes this is coming back as a string, so we force it into a tuple
            config["pointers"] = tuple(self._pointers)

        # If not signature is set, we assume this is where the Module was created and we're local to its code.
        # We'll collect the signatures of all the methods here before saving or sending them somewhere.
        # Note that we even do this for built-in modules, because 1) we want their methods preserved in Den for when
        # they're called via HTTP and 2) we want to preserve the exact set of methods in case the methods on built-in
        # modules change across Runhouse versions.
        config["signature"] = self.signature(rich=True)

        # Only save the endpoint if it's present in _endpoint or externally accessible
        config["endpoint"] = self.endpoint(external=True)

        if self._openapi_spec:
            config["openapi_spec"] = self._openapi_spec

        return config

    @classmethod
    def from_config(cls, config: dict, dryrun=False):
        if config.get("pointers"):
            config.pop("resource_subtype", None)
            logger.debug(f"Constructing module from pointers {config['pointers']}")
            (module_path, module_name, class_name) = config["pointers"]
            try:
                module_cls = cls._get_obj_from_pointers(
                    module_path, module_name, class_name
                )
                if not issubclass(module_cls, Module):
                    # Case when module was created through rh.module(new_class) factory, and needs to be
                    # made into a subclass of rh.Module. We'll follow the same flow as the subclass-created module
                    # below, where we don't call __init__ explicitly, because __init__ will call the subclass's init
                    # and this may a "type" module rather than an "instance". The user might instantiate it later, or
                    # it may be populated with attributes by the servlet's put_resource.
                    module_cls = _module_subclass_factory(
                        module_cls, config.get("pointers")
                    )
            except (ModuleNotFoundError, ImportError, AttributeError) as e:
                if (isinstance(e, ModuleNotFoundError) and module_name in str(e)) or (
                    (isinstance(e, ImportError) and class_name in str(e))
                    or (isinstance(e, AttributeError) and class_name in str(e))
                ):
                    system = config.get("system", None)
                    env = config.get("env", None)
                    env_name = (
                        env
                        if isinstance(env, str)
                        else env["name"]
                        if isinstance(env, Dict)
                        else "base_env"
                        if env is not None
                        else None
                    )
                    if (system and _current_cluster == _get_cluster_from(system)) and (
                        env_name
                        and obj_store.cluster_servlet
                        and obj_store.cluster_servlet.name == env_name
                    ):
                        # Could not load Module locally from within the system where it lives
                        raise e

                    # If we fail to construct the module class from the pointers, but system is elsewhere,
                    # we can still use this module from its signature alone.
                    module_cls = Module
                else:
                    # Other type of module or import error, such as dependency not installed
                    raise e

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
            new_module.access_level = config.pop("access_level", ResourceAccess.WRITE)
            new_module.visibility = config.pop("visibility", ResourceVisibility.PRIVATE)
            new_module._endpoint = config.pop("endpoint", None)
            new_module._pointers = config.pop("pointers", None)
            new_module._signature = config.pop("signature", None)
            new_module.dryrun = config.pop("dryrun", False)
            new_module.provenance = config.pop("provenance", None)
            new_module._openapi_spec = config.pop("openapi_spec", None)
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
    def _check_for_child_configs(cls, config: dict):
        """Overload by child resources to load any resources they hold internally."""
        system = config.get("system")
        if isinstance(system, str) or isinstance(system, dict):
            config["system"] = _get_cluster_from(system)
        env = config.get("env")
        if isinstance(env, str) or isinstance(env, dict):
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

    def _compute_signature(self, rich=False):
        return {
            name: self.method_signature(method) if rich else None
            for (name, method) in inspect.getmembers(self.__class__)
            if not name[0] == "_"
            and name not in MODULE_ATTRS
            and name not in MODULE_METHODS
            and callable(method)
            # Checks if there's an arg called "local" in the method signature, and if so, if it's default is True.
            and not getattr(
                inspect.signature(method).parameters.get("local"), "default", False
            )
        }

    def signature(self, rich=False):
        if self._signature:
            return self._signature

        if not rich:
            if self._dumb_signature_cache is None:
                self._dumb_signature_cache = self._compute_signature(rich=False)
            return self._dumb_signature_cache

        self._signature = self._compute_signature(rich=True)
        return self._signature

    def method_signature(self, method):
        """Extracts the properties of a method that we want to preserve when sending the method over the wire."""
        signature = inspect.signature(method)
        signature_metadata = {
            "signature": str(signature),
            "async": inspect.iscoroutinefunction(method)
            or inspect.isasyncgenfunction(method),
            "gen": inspect.isgeneratorfunction(method)
            or inspect.isasyncgenfunction(method),
            "doc": inspect.getdoc(method) if method.__doc__ else None,
            "annotations": str(inspect.getfullargspec(method).annotations),
        }

        return signature_metadata

    def _signature_extensions(self, item):
        """A way to manually extend the methods behind names in the signature. We can't use __getattr__
        because it gets too crazy and circular with the overloaded __getattribute__ below."""
        return None

    def endpoint(self, external: bool = False):
        """The endpoint of the module on the cluster. Returns an endpoint if one was manually set (e.g. if loaded
        down from a config). If not, request the endpoint from the Module's system.

        Args:
            external: If True and getting the endpoint from the system, only return an endpoint if it's externally
                accessible (i.e. not on localhost, not connected through as ssh tunnel). If False, return the endpoint
                even if it's not externally accessible.
        """
        if self._endpoint:
            return self._endpoint

        if (
            self._system
            and hasattr(self._system, "endpoint")
            and self._system.endpoint(external=external)
        ):
            return f"{self._system.endpoint(external=external)}/{self.name}"

        return None

    def set_endpoint(self, new_endpoint: str):
        self._endpoint = new_endpoint

    def _client(self):
        """Return the client through which to interact with the source-of-truth Module. If this module is local,
        i.e. this module is its own source of truth, return None."""
        if (
            hasattr(self, "_system")
            and self._system
            and hasattr(self._system, "_client")
        ):
            return self._system._client()
        if (
            hasattr(self, "_endpoint")
            and self._endpoint
            and isinstance(self._endpoint, str)
        ):
            return HTTPClient.from_endpoint(
                self._endpoint, resource_address=self.rns_address
            )
        return None

    def _remote_init(self, *args, **kwargs):
        if len(self.__class__.__bases__) > 1:
            module_cls = self.__class__.__bases__[1]
        elif self._pointers:
            (module_path, module_name, class_name) = self._pointers
            # Reload needs to be false here, because if we reload the class, the reloaded class actually doesn't
            # match the class object of which self was a subclass, so we can't call super().__init__ on it within
            # module_cls's __init__.
            # We'd get "TypeError: super(type, obj): obj must be an instance or subtype of type"
            module_cls = self._get_obj_from_pointers(
                module_path, module_name, class_name, reload=False
            )
        else:
            module_cls = self.__class__

        # Change around the MRO so that the module_cls is the first parent class, and Module is the second,
        # so methods like .to default to the module_cls and not Module's when on the cluster.
        # This is a small price to pay for matching PyTorch's .to API. If it creates too much craziness we can
        # revisit.
        # class NewSubclass(module_cls, Module):
        #     pass
        #
        # self.__class__ = NewSubclass

        module_cls.__init__(self, *args, **kwargs)

    @staticmethod
    def _get_obj_from_pointers(module_path, module_name, obj_name, reload=True):
        """Helper method to load a class or function from a module path, module name, and class name."""
        if module_path:
            abs_path = str((Path.home() / module_path).expanduser().resolve())
            sys.path.insert(0, abs_path)
            logger.debug(f"Appending {module_path} to sys.path")

        if module_name in obj_store.imported_modules and reload:
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

    def _extract_state(self):
        # Exclude anything already being sent in the config and private module attributes
        state = {}
        # We only send over state for instances, not classes
        if not isinstance(self, type):
            state = {
                attr: val
                for attr, val in self.__dict__.items()
                if attr not in MODULE_ATTRS and attr not in MODULE_METHODS
            }
        return state

    def default_name(self):
        return (
            self._pointers[2] if self._pointers else None
        ) or _generate_default_name(prefix=self.__class__.__qualname__.lower())

    def to(
        self,
        system: Union[str, Cluster],
        env: Optional[Union[str, List[str], Env]] = None,
        name: Optional[str] = None,
        force_install: bool = False,
    ):
        """Put a copy of the module on the destination system and env, and return the new module.

        Example:
            >>> local_module = rh.module(my_class)
            >>> cluster_module = local_module.to("my_cluster")
        """
        if system == self.system and env == self.env:
            if name and not self.name == name:
                # TODO return duplicate object under new name, don't rename
                self.rename(name)
            return self

        if system == "here":
            from runhouse import here

            system = here

        if system == "file":
            raise ValueError(
                f"Need an initialized local server in order to put {self} onto `rh.here`."
            )

        system = (
            _get_cluster_from(system, dryrun=self.dryrun) if system else self.system
        )
        env = env or self.env
        env = _get_env_from(env)

        if system:
            system.check_server()
            if env:
                env = env.to(system, force_install=force_install)

        # We need to backup the system here so the __getstate__ method of the cluster
        # doesn't wipe the client of this function's cluster when deepcopy copies it.
        hw_backup = self.system
        self.system = None
        new_module = copy.copy(self)
        self.system = hw_backup

        new_module.system = system
        new_module.env = env
        new_module.dryrun = True

        if isinstance(system, Cluster):
            new_name = name or self.name or self.default_name()
            if self.rns_address:
                new_name = f"{self._rns_folder}/{new_name}"
            new_module.name = new_name
            # TODO dedup with _extract_state
            # Exclude anything already being sent in the config and private module attributes
            excluded_state_keys = list(new_module.config().keys()) + [
                "_system",
                "_name",
                "_rns_folder",
                "dryrun",
                "_env",
                "_pointers",
                "_resolve",
            ]
            state = {}
            # We only send over state for instances, not classes
            if not isinstance(self, type):
                state = {
                    attr: val
                    for attr, val in self.__dict__.items()
                    if attr not in excluded_state_keys
                }
            logger.info(
                f"Sending module {new_module.name} to {system.name or 'local Runhouse daemon'}"
            )
            system.put_resource(new_module, state, dryrun=True)

        return new_module

    def get_or_to(
        self,
        system: Union[str, Cluster],
        env: Optional[Union[str, List[str], Env]] = None,
        name: Optional[str] = None,
    ):
        """Check if the module already exists on the cluster, and if so return the module object.
        If not, put the module on the cluster and return the remote module.

        Example:
            >>> remote_df = Model().get_or_to(my_cluster, name="remote_model")
        """
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
        if (
            item in MODULE_METHODS
            or item in MODULE_ATTRS
            or not hasattr(self, "_client")
        ):
            return super().__getattribute__(item)

        try:
            name = super().__getattribute__("_name")
        except AttributeError:
            return super().__getattribute__(item)

        if item not in self.signature(rich=False) or not name:
            return super().__getattribute__(item)

        # Do this after the signature and name check because it's potentially expensive
        client = super().__getattribute__("_client")()

        if not client:
            return super().__getattribute__(item)

        is_coroutine_function = (
            self.signature(rich=True)[item]["async"]
            and not self.signature(rich=True)[item]["gen"]
        )

        class RemoteMethodWrapper:
            """Helper class to allow methods to be called with __call__, remote, or run."""

            def __call__(self, *args, **kwargs):
                # stream_logs and run_name are both supported args here, but we can't include them explicitly because
                # the local code path here will throw an error if they are included and not supported in the
                # method signature.

                # Always take the user overrided behavior if they want it
                if "run_async" in kwargs:
                    run_async = kwargs.pop("run_async")
                else:
                    run_async = is_coroutine_function

                if run_async:
                    return client.acall(
                        name,
                        item,
                        run_name=kwargs.pop("run_name", None),
                        stream_logs=kwargs.pop("stream_logs", True),
                        remote=kwargs.pop("remote", False),
                        data={"args": args, "kwargs": kwargs},
                    )
                else:
                    return client.call(
                        name,
                        item,
                        run_name=kwargs.pop("run_name", None),
                        stream_logs=kwargs.pop("stream_logs", True),
                        remote=kwargs.pop("remote", False),
                        data={"args": args, "kwargs": kwargs},
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
                    **kwargs,
                )

        return RemoteMethodWrapper()

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

    def replicate(self, num_replicas=1, names=None, envs=None, parallel=False):
        """Replicate the module on the cluster in a new env and return the new modules."""
        if not self.system or not self.name:
            raise ValueError(
                "Cannot replicate a module that is not on a cluster. Please send the module to a cluster first."
            )
        if not isinstance(self.system, Cluster):
            raise ValueError(
                "Cannot replicate a module that is not on a cluster. Please send the module to a cluster first."
            )
        if envs and not len(envs) == num_replicas:
            raise ValueError(
                "If envs is a list, it must be the same length as num_replicas."
            )
        if names and not len(names) == num_replicas:
            raise ValueError(
                "If names is a list, it must be the same length as num_replicas."
            )

        def create_replica(i):
            name = names[i] if isinstance(names, list) else f"{self.name}_replica_{i}"

            if isinstance(envs, list):
                env = _get_env_from(envs[i])
            else:
                # We do a shallow copy here because we want to reuse the Package objects in the env
                # If we reconstruct from scratch, the cluster may not have been saved (but has a name), and won't
                # be populated properly inside the package's folder's system.
                env = copy.copy(self.env)
                env.name = f"{self.env.name}_replica_{i}"

            new_module = copy.copy(self)
            new_module.name = name
            new_module.env = None
            new_module.system = None
            new_module = new_module.to(self.system, env=env)
            return new_module

        if parallel:
            from multiprocessing import pool

            with pool.ThreadPool() as p:
                return list(p.imap(create_replica, range(num_replicas)))

        return [create_replica(i) for i in range(num_replicas)]

    @property
    def remote(self):
        """Helper property to allow for access to remote properties, both public and private. Returning functions
        is not advised.

        Example:
            >>> my_module.remote.my_property
            >>> my_module.remote._my_private_property
            >>> my_module.remote.size = 14
        """
        client = super().__getattribute__("_client")()
        system = super().__getattribute__("_system")
        name = super().__getattribute__("_name")

        outer_super_gettattr = super().__getattribute__
        outer_super_setattr = super().__setattr__

        class RemotePropertyWrapper:
            @classmethod
            def __getattribute__(cls, item):
                # TODO[DG] revise
                if not client or not name:
                    return outer_super_gettattr(item)

                return client.call(name, item, stream_logs=False)

            @classmethod
            def __setattr__(cls, key, value):
                if not client or not name:
                    return outer_super_setattr(key, value)

                return client.call(
                    key=name,
                    method_name=key,
                    data={"args": [value], "kwargs": {}},
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

    def fetch(self, item: str = None, **kwargs):
        """Helper method to allow for access to remote state, both public and private. Fetching functions
        is not advised. `system.get(module.name).resolved_state()` is roughly equivalent to `module.fetch()`.

        Example:
            >>> my_module.fetch("my_property")
            >>> my_module.fetch("my_private_property")

            >>> MyRemoteClass = rh.module(my_class).to(system)
            >>> MyRemoteClass(*args).fetch() # Returns a my_class instance, populated with the remote state

            >>> my_blob.fetch() # Returns the data of the blob, due to overloaded ``resolved_state`` method

            >>> class MyModule(rh.Module):
            >>>     # ...
            >>>
            >>> MyModule(*args).to(system).fetch() # Returns the full remote module, including private and public state
        """
        name = super().__getattribute__("_name")

        client = super().__getattribute__("_client")()

        if item is not None:
            if not client or not name:
                return super().__getattribute__(item)
            return client.call(name, item)
        else:
            if not client or not name:
                return self.resolved_state(**kwargs)
            return client.call(
                name, "resolved_state", data={"args": [], "kwargs": kwargs}
            )

    async def fetch_async(
        self, key: str, remote: bool = False, stream_logs: bool = False
    ):
        """Async version of fetch. Can't be a property like `fetch` because __getattr__ can't be awaited.

        Example:
            >>> await my_module.fetch_async("my_property")
            >>> await my_module.fetch_async("_my_private_property")
        """
        client = super().__getattribute__("_client")()
        system = super().__getattribute__("_system")
        name = super().__getattribute__("_name")

        def call_wrapper():
            if not key:
                return client.get(name, remote=remote)

            if isinstance(system, Cluster) and name and system.on_this_cluster():
                obj_store_obj = obj_store.get(name, default=None)
                if obj_store_obj:
                    return obj_store_obj.__getattribute__(key)
                else:
                    return self.__getattribute__(key)
            return client.call(name, key, remote=remote)

        try:
            is_gen = (key and hasattr(self, key)) and inspect.isasyncgenfunction(
                super().__getattribute__(key)
            )
        except AttributeError:
            is_gen = self.signature(rich=False).get(key, {}).get("gen", False)

        if is_gen:

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
        client = super().__getattribute__("_client")()
        if not client or not self._name:
            return super().__setattr__(key, value)

        def call_wrapper():
            return self._client().call(
                key=self._name,
                method_name=key,
                data={"args": [value], "kwargs": {}},
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
        Runhouse-unaware function. For example, if you call a Runhouse-unaware function with ``.remote()``,
        you will be returned a Blob which wraps your data. If you want to pass that Blob into another function
        that operates on the original data (e.g. a function that takes a numpy array), you can call
        ``my_second_fn(my_blob.resolve())``, and ``my_blob`` will be replaced with the contents of its ``.data`` on the
        cluster before being passed into ``my_second_fn``.

        Resolved state is defined by the ``resolved_state`` method. By default, modules created with the
        ``rh.module`` factory constructor will be resolved to their original non-module-wrapped class (or best attempt).
        Modules which are defined as a subclass of ``Module`` will be returned as-is, as they have no other
        "original class."

        Example:
            >>> my_module = rh.module(my_class)
            >>> my_remote_fn(my_module.resolve()) # my_module will be replaced with the original class `my_class`

            >>> my_result_blob = my_remote_fn.call.remote(args)
            >>> my_other_remote_fn(my_result_blob.resolve()) # my_result_blob will be replaced with its data

        """
        self._resolve = True
        return self

    def resolved_state(self):
        """Return the resolved state of the module. By default, this is the original class of the module if it was
        created with the ``module`` factory constructor."""
        if not self._pointers:
            self._resolve = False
            return self

        (module_path, module_name, class_name) = self._pointers
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

    def _save_sub_resources(self, folder: str = None):
        if isinstance(self.system, Resource) and self.system.name:
            self.system.save(folder=folder)
        if isinstance(self.env, Resource) and self.env.name != Env.DEFAULT_NAME:
            self.env.save(folder=folder)

    def rename(self, name: str):
        """Rename the module."""
        if self.name == name or self.rns_address == name:
            return
        old_name = self.name
        self.name = name  # Goes through Resource setter to parse name properly (e.g. if rns path)
        if (
            self.system
            and isinstance(self.system, Cluster)
            and self.system.on_this_cluster()
        ):
            # We rename the object with the full rns_address here because if it's a resource, we want the
            # object stored in the obj store to have the updated rns_address, not just the updated key.
            obj_store.rename(old_key=old_name, new_key=self.rns_address or self.name)
        elif self._client():
            self._client().rename(
                old_key=old_name, new_key=self.rns_address or self.name
            )

    def save(self, name: str = None, overwrite: bool = True, folder: str = None):
        """Register the resource and save to local working_dir config and RNS config store."""
        # Need to override Resource's save to handle key changes in the obj store
        # Also check that this is a Blob and not a File

        # Make sure to generate the openapi spec before saving
        _ = self.openapi_spec()

        old_rns_address = self.rns_address
        if name:
            _, base_name = rns_client.split_rns_name_and_path(
                rns_client.resolve_rns_path(name)
            )
            if self.name != base_name:
                if overwrite:
                    self.rename(name)
                else:
                    self.name = name
                    if isinstance(self.system, Cluster):
                        self.system.put_resource(self)

        res = super().save(overwrite=overwrite, folder=folder)

        if old_rns_address != self.rns_address:
            self.remote.name = self.rns_address

        return res

    def share(self, *args, visibility=None, **kwargs):
        if visibility and not visibility == self.visibility:
            self.visibility = visibility
            self.remote.visibility = (
                visibility  # Sets the visibility on the remote resource
            )
        return super().share(*args, **kwargs, visibility=visibility)

    @staticmethod
    def _extract_module_path(raw_cls_or_fn: Union[Type, Callable]):
        py_module = inspect.getmodule(raw_cls_or_fn)

        # Need to resolve in case just filename is given
        module_path = (
            str(Path(inspect.getfile(py_module)).resolve())
            if hasattr(py_module, "__file__")
            else None
        )

        return module_path

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
        module_path = Module._extract_module_path(raw_cls_or_fn)

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
                    logger.debug(f"Module files: {module_path}")
                    logger.debug(f"Package paths: {package_paths}")
                    logger.debug(f"Base dirs: {base_dirs}")
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

    def openapi_spec(self, spec_name: Optional[str] = None):
        """Generate an OpenAPI spec for the module.

        TODO: This breaks if the module has type annotations that are classes, and not standard library or
        typing types.

        Maybe we can do something using: https://github.com/kuimono/openapi-schema-pydantic to allow
        nested Pydantic models easily as schemas?

        TODO: What happens if there is an empty function, will this work with an empty body even though it is
        marked as required?

        """
        if self._openapi_spec is not None:
            return self._openapi_spec

        spec_name = spec_name or self.name

        if not spec_name:
            raise ValueError(
                "Module must have a name to generate an OpenAPI spec. Try saving the module first."
            )

        spec = APISpec(
            title=spec_name,
            version="1.0.0",
            openapi_version="3.0.0",
            plugins=[],
        )

        base_module_class_methods = {
            m[0] for m in inspect.getmembers(Module, predicate=inspect.isfunction)
        }
        for method_name, method in inspect.getmembers(
            self.__class__, predicate=inspect.isfunction
        ):
            if method_name.startswith("_") or method_name in base_module_class_methods:
                continue

            spec.path(
                path=f"/{spec_name}/{method_name}",
                operations={
                    "post": {
                        "summary": method.__doc__.strip().split("\n")[0]
                        if method.__doc__
                        else "",
                        "description": method.__doc__.strip() if method.__doc__ else "",
                        "requestBody": {
                            "content": {
                                "application/json": {
                                    "schema": f"{method_name}_body_schema",
                                },
                            },
                            "required": True,
                        },
                        "responses": {
                            "200": {
                                "description": "A response containing an output_type, data, and serialization, and optionally an error and traceback."
                            },
                        },
                    },
                },
            )

            # Generate schema for the Request Body
            params = {}
            for param_name, param in inspect.signature(method).parameters.items():
                if param_name == "self":
                    continue

                if param.annotation and param.annotation != inspect.Parameter.empty:
                    param_type = (
                        param.annotation.__name__
                        if isinstance(param.annotation, type)
                        else param.annotation
                    )
                else:
                    param_type = str

                params[param_name] = (
                    param_type,
                    param.default if param.default != inspect.Parameter.empty else ...,
                )

            module_method_params = create_model(
                f"{method_name}_schema", **params
            ).schema()
            module_method_params["title"] = "kwargs"

            request_body_schema = CallParams.schema()
            request_body_schema["properties"]["data"] = {
                "title": "data",
                "type": "object",
                "properties": {
                    "kwargs": module_method_params,
                },
                "required": ["kwargs"],
            }
            request_body_schema["title"] = f"{method_name} Params"
            spec.components.schema(f"{method_name}_body_schema", request_body_schema)

        self._openapi_spec = spec.to_dict()
        return self._openapi_spec

    # Found in python decorator logic, maybe use
    # func_name = getattr(f, '__qualname__', f.__name__)
    # module_name = getattr(f, '__module__', '')
    # if module_name:
    #     full_name = f'{module_name}.{func_name}'
    # else:
    #     full_name = func_name


def _module_subclass_factory(cls, cls_pointers):
    def __init__(
        self,
        *args,
        system=None,
        env=None,
        dryrun=False,
        pointers=cls_pointers,
        signature=None,
        name=None,
        provenance=None,
        **kwargs,
    ):
        # args and kwargs are passed to the cls's __init__ method if this is being called on a cluster. They
        # shouldn't be passed otherwise.
        Module.__init__(
            self,
            pointers=pointers,
            signature=signature,
            name=name,
            system=system,
            env=env,
            dryrun=dryrun,
            provenance=provenance,
        )
        # This allows a class which is already on the cluster to construct an instance of itself with a factory
        # method, e.g. my_module = MyModuleCls.factory_constructor(*args, **kwargs)
        if self.system and self.system.on_this_cluster():
            self._remote_init(*args, **kwargs)

    @classmethod
    def _module_init_only(cls, *args, **kwargs):
        new_module = cls.__new__(cls)
        Module.__init__(new_module, *args, **kwargs)
        return new_module

    def __call__(
        self,
        *args,
        dryrun=False,
        name=None,
        **kwargs,
    ):
        new_module = copy.copy(self)
        # Create a copy of the item on the cluster under the new name
        new_module.name = name or self.name
        new_module.dryrun = dryrun
        env = _get_env_from(kwargs.get("env") or self.env)
        if not new_module.dryrun and new_module.system:
            # We use system.put_resource here because the signatures for HTTPClient.put_resource and
            # obj_store.put_resource are different, but we should fix that.
            new_module.system.put_resource(new_module, env=env)
            new_module.system.call(new_module.name, "_remote_init", *args, **kwargs)
        else:
            new_module._remote_init(*args, **kwargs)

        return new_module

    methods = {
        "__init__": __init__,
        "__call__": __call__,
        "_module_init_only": _module_init_only,
    }
    new_type = type(cls_pointers[2], (Module, cls), methods)
    return new_type


def module(
    cls: [Type] = None,
    name: Optional[str] = None,
    system: Optional[Union[str, Cluster]] = None,  # deprecated
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
        env (Optional[str or Env]): Environment in which the module should live on the cluster, if system is cluster.
        dryrun (bool): Whether to create the Blob if it doesn't exist, or load a Blob object as a dryrun.
            (Default: ``False``)

    Returns:
        Module: The resulting module.

    Example - creating a module by defining an rh.Module subclass:
        >>> import runhouse as rh
        >>> import transformers
        >>>
        >>> # Sample rh.Module class
        >>> class Model(rh.Module):
        >>>    def __init__(self, model_id, device="cpu", env=None):
        >>>        # Note that the code here will be run in your local environment prior to being sent to
        >>>        # to a cluster. For loading large models/datasets that are only meant to be used remotely,
        >>>        # we recommend using lazy initialization (see tokenizer and model attributes below).
        >>>        super().__init__(env=env)
        >>>        self.model_id = model_id
        >>>        self.device = device
        >>>
        >>>    @property
        >>>    def tokenizer(self):
        >>>        # Lazily initialize the tokenizer remotely only when it is needed
        >>>        if not hasattr(self, '_tokenizer'):
        >>>            self._tokenizer = transformers.AutoTokenizer.from_pretrained(self.model_id)
        >>>        return self._tokenizer
        >>>
        >>>    @property
        >>>    def model(self):
        >>>        if not hasattr(self, '_model'):
        >>>            self._model = transformers.AutoModel.from_pretrained(self.model_id).to(self.device)
        >>>        return self._model
        >>>
        >>>    def predict(self, x):
        >>>        x = self.tokenizer(x, return_tensors="pt")
        >>>        return self.model(x)

        >>> # Creating rh.Module instance
        >>> model = Model(model_id="bert-base-uncased", device="cuda", env="my_env")
        >>> model = model.to(system="my_gpu")
        >>> model.predict("Hello world!")   # Runs on system in env
        >>> tok = model.remote.tokenizer    # Returns remote tokenizer
        >>> id = model.local.model_id       # Returns local model_id, if any
        >>> model_id = model.model_id       # Returns local model_id (not remote)
        >>> model.fetch()                   # Returns full remote module, including model and tokenizer
        >>>

    Example - creating a Module from an existing class, via the rh.module() factory method:
        >>> other_model = Model(model_id="bert-base-uncased", device="cuda").to("my_gpu", "my_env")
        >>>
        >>> # Another method: Create a module instance from an existing non-Module class using rh.module()
        >>> RemoteModel = rh.module(cls=BERTModel, env="my_env")
        >>> remote_model = RemoteModel(model_id="bert-base-uncased", device="cuda").to(system="my_gpu")
        >>> remote_model.predict("Hello world!")  # Runs on system in env
        >>>
        >>> # You can also call remote class methods
        >>> other_model = RemoteModel.get_model_size("bert-base-uncased")

        >>> # Loading a module
        >>> my_local_module = rh.module(name="~/my_module")
        >>> my_s3_module = rh.module(name="@/my_module")
    """
    if name and not any([cls, system, env]):
        # Try reloading existing module
        return Module.from_name(name, dryrun)

    if system:
        raise Exception(
            "`system` argument is no longer supported in the module factory function. "
            "Use `.to(system)` or `.get_or_to(system)` after construction to send and run the Module on the system."
        )

    if not isinstance(env, Env):
        env = _get_env_from(env) or Env(name=Env.DEFAULT_NAME)
        env.working_dir = env.working_dir or "./"

    cls_pointers = Module._extract_pointers(cls, env.reqs)
    name = name or (
        cls_pointers[2] if cls_pointers else _generate_default_name(prefix="module")
    )

    module_subclass = _module_subclass_factory(cls, cls_pointers)
    return module_subclass._module_init_only(
        env=env,
        dryrun=dryrun,
        pointers=cls_pointers,
        name=name,
    )


MODULE_METHODS = dir(Module)
