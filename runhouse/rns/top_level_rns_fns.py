import logging
import sys
from typing import List

from runhouse.logger import LOGGING_CONFIG
from runhouse.rh_config import configs, rns_client
from runhouse.rns.utils.hardware import _get_cluster_from

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


# This funny structure lets us to rh.here to get the current cluster
def __getattr__(name):
    if name == "here":
        from runhouse.rns.utils.hardware import _current_cluster, _get_cluster_from

        config = _current_cluster(key="config")
        if not config:
            return "file"
        system = _get_cluster_from(config)
        return system
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


def load_from_path(
    path: str,
    instantiate: bool = True,
):
    pass


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
    import subprocess

    subprocess.Popen("pip install ipython".split(" "))
    # TODO install ipython if not installed
    import IPython

    IPython.embed()


def delete(resource_or_name: str):
    """Delete the resource from the RNS or local config store."""
    rns_client.delete_configs(resource=resource_or_name)


# TODO [DG]
def delete_all(folder: str = None):
    """Delete all resources in the given folder, such that the user has peace of mind that they are not consuming
    any hidden cloud costs."""
    pass


# TODO [DG]
def sync_down():
    pass


# TODO [DG]
def load_all_clusters():
    """Load all clusters in RNS into the local Sky context."""
    pass


# -----------------  Pinning objects to cluster memory  -----------------
from runhouse import rh_config


def pin_to_memory(key: str, value):
    # Put the obj_ref in Ray obj store here so it doesn't need to be deserialized inside the ObjStoreActor's process
    # which may not have the necessary modules installed or gpu access
    # obj_ref = ray.put(value)
    # rh_config.obj_store.put_obj_ref(key, obj_ref)
    rh_config.obj_store.put(key, value)


def get_pinned_object(key: str, default=None):
    # ref = rh_config.obj_store.get_obj_ref(key)
    # if ref is None:
    #     return default
    # else:
    #     return ray.get(ref)
    return rh_config.obj_store.get(key, default=default)


def get(key: str, cluster=None, default=None):
    system = _get_cluster_from(cluster)
    if system:
        return system.get(key, default=default)
    return rh_config.obj_store.get(key, default=default)


def remove_pinned_object(key: str):
    rh_config.obj_store.delete(key)


def pop_pinned_object(key: str, default=None):
    return rh_config.obj_store.pop(key, default=default)


def pinned_keys():
    return rh_config.obj_store.keys()


def clear_pinned_memory():
    rh_config.obj_store.clear()
