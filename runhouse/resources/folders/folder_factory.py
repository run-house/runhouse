import logging
from pathlib import Path
from typing import Dict, Optional, Union

import fsspec

from runhouse.resources.folders.folder import Folder
from runhouse.resources.hardware.utils import _get_cluster_from
from runhouse.resources.resource import Resource

logger = logging.getLogger(__name__)


def folder(
    name: Optional[str] = None,
    path: Optional[Union[str, Path]] = None,
    system: Optional[Union[str, "Cluster"]] = None,
    dryrun: bool = False,
    local_mount: bool = False,
    data_config: Optional[Dict] = None,
) -> Folder:
    """Creates a Runhouse folder object, which can be used to interact with the folder at the given path.

    Args:
        name (Optional[str]): Name to give the folder, to be re-used later on.
        path (Optional[str or Path]): Path (or path) that the folder is located at.
        system (Optional[str or Cluster]): File system or cluster name. If providing a file system this must be one of:
            [``file``, ``github``, ``sftp``, ``ssh``, ``s3``, ``gs``, ``azure``].
            We are working to add additional file system support.
        dryrun (bool): Whether to create the Folder if it doesn't exist, or load a Folder object as a dryrun.
            (Default: ``False``)
        local_mount (bool): Whether or not to mount the folder locally. (Default: ``False``)
        data_config (Optional[Dict]): The data config to pass to the underlying fsspec handler.

    Returns:
        Folder: The resulting folder.

    Example:
        >>> import runhouse as rh
        >>> rh.folder(name='training_imgs', path='remote_directory/images', system='s3').save()

        >>> # Load folder from above
        >>> reloaded_folder = rh.folder(name="training_imgs")
    """
    # TODO [DG] Include loud warning that relative paths are relative to the git root / working directory!

    if name and not any([path, system, local_mount, data_config]):
        # If only the name is provided
        try:
            return Folder.from_name(name, dryrun)
        except ValueError:
            # This is a rare instance where passing no constructor params is actually valid (anonymous folder),
            # so if we don't find the name, the user may still actually want to create a new folder.
            pass

    system = system or Folder.DEFAULT_FS
    if system == "s3":
        from .s3_folder import S3Folder

        return S3Folder(
            system=system,
            path=path,
            data_config=data_config,
            local_mount=local_mount,
            name=name,
            dryrun=dryrun,
        )
    elif system == "gs":
        from .gcs_folder import GCSFolder

        return GCSFolder(
            system=system,
            path=path,
            data_config=data_config,
            local_mount=local_mount,
            name=name,
            dryrun=dryrun,
        )
    elif system == "azure":
        from .azure_folder import AzureFolder

        return AzureFolder(
            system=system,
            path=path,
            data_config=data_config,
            local_mount=local_mount,
            name=name,
            dryrun=dryrun,
        )

    if system in fsspec.available_protocols():
        if system not in [Folder.DEFAULT_FS, "github", "sftp", "ssh"]:
            logger.warning(
                f"fsspec file system {system} not officially supported. Use at your own risk."
            )
        return Folder(
            system=system,
            path=path,
            data_config=data_config,
            local_mount=local_mount,
            name=name,
            dryrun=dryrun,
        )

    cluster_system = _get_cluster_from(system, dryrun=dryrun)
    if isinstance(cluster_system, Resource):
        return Folder(
            system=cluster_system,
            path=path,
            data_config=data_config,
            local_mount=local_mount,
            name=name,
            dryrun=dryrun,
        )

    raise ValueError(
        f"File system {system} not found. Have you installed the "
        f"necessary packages for this fsspec protocol? (e.g. s3fs for s3). If the file system "
        f"is a cluster (ex: /my-user/rh-cpu), make sure the cluster config has been saved."
    )
