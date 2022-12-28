import sys
from typing import Optional, List

from runhouse.rh_config import rns_client


def resolve_rns_path(path: str):
    return rns_client.resolve_rns_path(path)


def exists(name,
           resource_type: str = None,
           load_from: Optional[List[str]] = None):
    return rns_client.exists(name, resource_type=resource_type, load_from=load_from)


def locate(name_or_path,
           resolve_path: bool = True,
           load_from: Optional[List[str]] = None):
    return rns_client.locate(name_or_path,
                             resolve_path=resolve_path,
                             load_from=load_from
                             )


def load(name: str,
         load_from: Optional[List[str]] = None,
         instantiate: bool = True,
         dryrun: bool = False):
    config = rns_client.load_config(name=name, load_from=load_from)
    if not instantiate:
        return config
    from_config_constructor = getattr(sys.modules['runhouse.rns'], config['resource_type'].capitalize(), None).from_config
    if not from_config_constructor:
        raise ValueError(f"Could not find constructor for type {config['resource_type']}")
    return from_config_constructor(config=config, dryrun=dryrun)


def load_from_path(path: str,
                   instantiate: bool = True,
                   load_from: Optional[List[str]] = None,
                   ):
    pass


def set_save_to(save_to: List[str]):
    rns_client.save_to = save_to


def set_load_from(load_from: List[str]):
    rns_client.load_from = load_from


def save(resource,
         name: str = None,
         save_to: Optional[List[str]] = None,
         snapshot: bool = False,
         overwrite: bool = True,
         **snapshot_kwargs):  # TODO [DG] was this supposed to be kwargs for the snapshot?
    """Register the resource, saving it to local working_dir config and/or RNS config store. Uses the resource's
    `self.config_for_rns` to generate the dict to save."""

    # TODO handle self.access == 'read' instead of this weird overwrite argument
    snapshot_kwargs = snapshot_kwargs or {}
    resource_to_save = resource.snapshot(**snapshot_kwargs) if snapshot else resource
    resource_to_save._name = name if name is not None else resource_to_save._name
    rns_client.save_config(resource=resource_to_save,
                           save_to=save_to,
                           overwrite=overwrite)


def set_folder(path: str, create=False):
    rns_client.set_folder(path=path, create=create)


def unset_folder():
    """ Sort of like `cd -`, but with a full stack of the previous folder's set. Resets the
    current_folder to the previous one on the stack, the current_folder right before the
    current one was set. """
    rns_client.unset_folder()


def current_folder():
    return rns_client.current_folder


def split_rns_name_and_path(path: str):
    return rns_client.split_rns_name_and_path(path)


# TODO [DG] I don't think this name is intuitive, we should change it
def resources(path: str = None,
              full_paths=False):
    path = path or current_folder()
    import runhouse as rh
    return rh.folder(name=path, save_to=[]).resources(full_paths=full_paths)


def ipython():
    import subprocess
    subprocess.Popen('pip install ipython'.split(' '))
    # TODO install ipython if not installed
    import IPython
    IPython.embed()


# TODO [DG]
def delete_all(folder: str = None):
    """ Delete all resources in the given folder, such that the user has peace of mind that they are not consuming
    any hidden cloud costs. """
    pass


# TODO [DG]
def sync_down():
    pass


# TODO [DG]
def load_all_clusters():
    """ Load all clusters in RNS into the local Sky context. """
    pass


# -----------------  Pinning objects to cluster memory  -----------------
# TODO is this a bad idea?

from runhouse import rh_config


def _set_pinned_memory_store(store: dict):
    rh_config.global_pinned_memory_store = store


def pin_to_memory(key: str, value):
    if rh_config.global_pinned_memory_store is None:
        rh_config.global_pinned_memory_store = {}
    rh_config.global_pinned_memory_store[key] = value


def get_pinned_object(key: str):
    if rh_config.global_pinned_memory_store:
        return rh_config.global_pinned_memory_store.get(key, None)
    else:
        return None


def remove_pinned_object(key: str):
    if rh_config.global_pinned_memory_store:
        rh_config.global_pinned_memory_store.pop(key, None)


def pop_pinned_object(key: str):
    if rh_config.global_pinned_memory_store:
        return rh_config.global_pinned_memory_store.pop(key, None)
    else:
        return None


def flush_pinned_memory():
    if rh_config.global_pinned_memory_store:
        rh_config.global_pinned_memory_store.clear()
