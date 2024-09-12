from pathlib import Path

from runhouse.globals import configs, rns_client
from runhouse.resources.hardware.utils import _get_cluster_from
from runhouse.utils import generate_default_name

DEFAULT_LOCAL_FOLDER = f"{Path.cwd()}/"
DEFAULT_CLUSTER_FS_FOLDER = (
    ""  # Objects will land inside home directory when sent without a path
)
DEFAULT_BLOB_STORAGE_FOLDER = (
    configs.get("default_blob_storage_folder", "runhouse") + "/"
)


def _generate_default_path(cls, name, system):
    """Generate a default path for a data resource. Logic is as follows:
    1. If the system is a local file system, save to the current working directory
    2. If the system is a remote file system, save to the default cache folder
    3. If the system is a remote object store, save to the default object store folder
    """

    from runhouse.resources.hardware import Cluster

    system = _get_cluster_from(system)

    name = name or generate_default_name(prefix=cls.RESOURCE_TYPE)
    if system == rns_client.DEFAULT_FS or "here":
        base_folder = DEFAULT_LOCAL_FOLDER
    elif isinstance(system, Cluster):
        if system.on_this_cluster():
            base_folder = DEFAULT_LOCAL_FOLDER
        else:
            base_folder = DEFAULT_CLUSTER_FS_FOLDER
    else:
        base_folder = DEFAULT_BLOB_STORAGE_FOLDER
    return f"{base_folder}{cls.RESOURCE_TYPE}/{name}"
