from datetime import datetime
from pathlib import Path

from runhouse.globals import configs, rns_client
from runhouse.resources.hardware.utils import _get_cluster_from

DEFAULT_LOCAL_FOLDER = f"{Path.cwd()}/"
DEFAULT_CLUSTER_FS_FOLDER = (
    ""  # Objects will land inside home directory when sent without a path
)
DEFAULT_BLOB_STORAGE_FOLDER = (
    configs.get("default_blob_storage_folder", "runhouse") + "/"
)


def _generate_default_name(prefix: str = None, precision: str = "s", sep="_") -> str:
    """Name of the Run's parent folder which contains the Run's data (config, stdout, stderr, etc).
    If a name is provided, prepend that to the current timestamp to complete the folder name."""
    if precision == "d":
        timestamp_key = f"{datetime.now().strftime('%Y%m%d')}"
    elif precision == "s":
        timestamp_key = f"{datetime.now().strftime(f'%Y%m%d{sep}%H%M%S')}"
    elif precision == "ms":
        timestamp_key = f"{datetime.now().strftime(f'%Y%m%d{sep}%H%M%S_%f')}"
    if prefix is None:
        return timestamp_key
    return f"{prefix}{sep}{timestamp_key}"


def _generate_default_path(cls, name, system):
    """Generate a default path for a data resource. Logic is as follows:
    1. If the system is a local file system, save to the current working directory
    2. If the system is a remote file system, save to the default cache folder
    3. If the system is a remote object store, save to the default object store folder
    """

    from runhouse.resources.hardware import Cluster

    system = _get_cluster_from(system)

    name = name or _generate_default_name(prefix=cls.RESOURCE_TYPE)
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
