import logging
import sys
from typing import List

from runhouse.globals import configs, rns_client

from runhouse.logger import LOGGING_CONFIG

# Configure the logger once
logging.config.dictConfig(LOGGING_CONFIG)


disable_data_collection = configs.get("disable_data_collection", False)
if not disable_data_collection:
    import sentry_sdk

    sentry_sdk.init(
        dsn="https://93f64f9efc194d5bb66edc0693fde714@o4505521613307904.ingest.sentry.io/4505522385911808",
        environment="production",
        # Set traces_sample_rate to 1.0 to capture 100%
        # of transactions for performance monitoring.
        # We recommend adjusting this value in production,
        traces_sample_rate=1.0,
    )


def resolve_rns_path(path: str):
    return rns_client.resolve_rns_path(path)


def exists(name, resource_type: str = None):
    return rns_client.exists(name, resource_type=resource_type)


def locate(name_or_path, resolve_path: bool = True):
    return rns_client.locate(
        name_or_path,
        resolve_path=resolve_path,
    )


def load(name: str, instantiate: bool = True, dryrun: bool = False):
    config = rns_client.load_config(name=name)
    if not instantiate:
        return config
    resource_class = getattr(
        sys.modules["runhouse"], config["resource_type"].capitalize(), None
    )
    if not resource_class:
        raise TypeError(
            f"Could not find module associated with {config['resource_type']}"
        )

    try:
        loaded = resource_class.from_config(config=config, dryrun=dryrun)
        rns_client.add_upstream_resource(name)
        return loaded
    except:
        raise ValueError(
            f"Could not find constructor for type {config['resource_type']}"
        )


# This funny structure lets us use `rh.here` to get the current cluster
def __getattr__(name):
    if name == "here":
        from runhouse.resources.hardware.utils import (
            _current_cluster,
            _get_cluster_from,
        )

        config = _current_cluster(key="config")
        if not config:
            return "file"
        system = _get_cluster_from(config, dryrun=True)
        return system
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


def set_save_to(save_to: List[str]):
    rns_client.save_to = save_to


def set_load_from(load_from: List[str]):
    rns_client.load_from = load_from


def save(
    resource,
    name: str = None,
    overwrite: bool = True,
):
    """Register the resource, saving it to local working_dir config and/or RNS config store. Uses the resource's
    `self.config_for_rns` to generate the dict to save."""

    # TODO handle self.access == 'read' instead of this weird overwrite argument
    if name:
        if "/" in name[1:] or resource._rns_folder is None:
            (
                resource._name,
                resource._rns_folder,
            ) = split_rns_name_and_path(resolve_rns_path(name))
        else:
            resource._name = name
    rns_client.save_config(resource=resource, overwrite=overwrite)


def set_folder(path: str, create=False):
    rns_client.set_folder(path=path, create=create)


def unset_folder():
    """Sort of like `cd -`, but with a full stack of the previous folder's set. Resets the
    current_folder to the previous one on the stack, the current_folder right before the
    current one was set."""
    rns_client.unset_folder()


def current_folder():
    return rns_client.current_folder


def split_rns_name_and_path(path: str):
    return rns_client.split_rns_name_and_path(path)


def resources(path: str = None, full_paths=False):
    path = path or current_folder()
    return rns_client.contents(name_or_path=path, full_paths=full_paths)


def ipython():
    try:
        import IPython

        IPython.embed()
    except ImportError:
        import code

        code.interact(local=locals())


def delete(resource_or_name: str):
    """Delete the resource from the RNS or local config store."""
    rns_client.delete_configs(resource=resource_or_name)


# -----------------  Pinning objects to cluster memory  -----------------
from runhouse import globals


def pin_to_memory(key: str, value):
    # Deprecate after 0.0.12
    import warnings

    warnings.warn(
        "pin_to_memory is deprecated, use `rh.here.put` instead", DeprecationWarning
    )
    rh_config.obj_store.put(key, value)


def get_pinned_object(key: str, default=None):
    # Deprecate after 0.0.12
    import warnings

    warnings.warn(
        "get_pinned_object is deprecated, use `rh.here.get` instead", DeprecationWarning
    )
    return rh_config.obj_store.get(key, default=default)


def remove_pinned_object(key: str):
    # Deprecate after 0.0.12
    import warnings

    warnings.warn(
        "remove_pinned_object is deprecated, use `rh.here.delete` instead",
        DeprecationWarning,
    )
    globals.obj_store.delete(key)


def pinned_keys():
    # Deprecate after 0.0.12
    import warnings

    warnings.warn(
        "pinned_keys is deprecated, use `rh.here.keys` instead",
        DeprecationWarning,
    )
    return globals.obj_store.keys()


def clear_pinned_memory():
    # Deprecate after 0.0.12
    import warnings

    warnings.warn(
        "clear_pinned_memory is deprecated, use `rh.here.clear` instead",
        DeprecationWarning,
    )
    globals.obj_store.clear()
