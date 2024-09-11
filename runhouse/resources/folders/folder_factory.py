from pathlib import Path
from typing import Optional, Union

from runhouse.logger import get_logger

from runhouse.resources.folders.folder import Folder
from runhouse.resources.hardware.utils import _get_cluster_from

logger = get_logger(__name__)


def folder(
    name: Optional[str] = None,
    path: Optional[Union[str, Path]] = None,
    system: Optional[Union[str, "Cluster"]] = None,
    load_from_den: bool = True,
    dryrun: bool = False,
) -> Folder:
    """Creates a Runhouse folder object, which can be used to interact with the folder at the given path.

    Args:
        name (Optional[str]): Name to give the folder, to be re-used later on.
        path (Optional[str or Path]): Path (or path) that the folder is located at.
        system (Optional[str or Cluster]): File system or cluster name. If prpre-oviding a file system this must be one of:
            [``file``, ``s3``, ``gs``].
        load_from_den (bool): Whether to try loading the Folder resource from Den. (Default: ``True``)
        dryrun (bool): Whether to create the Folder if it doesn't exist, or load a Folder object as a dryrun.
            (Default: ``False``)

    Returns:
        Folder: The resulting folder.

    Example:
        >>> import runhouse as rh
        >>> rh.folder(name='training_imgs', path='remote_directory/images', system='s3').save()

        >>> # Load folder from above
        >>> reloaded_folder = rh.folder(name="training_imgs")
    """
    # TODO [DG] Include loud warning that relative paths are relative to the git root / working directory!

    if name and not any([path, system]):
        # If only the name is provided
        try:
            return Folder.from_name(name, load_from_den=load_from_den, dryrun=dryrun)
        except ValueError:
            # This is a rare instance where passing no constructor params is actually valid (anonymous folder),
            # so if we don't find the name, the user may still actually want to create a new folder.
            pass

    system = system or Folder.DEFAULT_FS
    if system == "s3":
        from .s3_folder import S3Folder

        logger.debug(f"Creating a S3 folder with name: {name}")
        return S3Folder(
            system=system,
            path=path,
            name=name,
            dryrun=dryrun,
        )
    elif system == "gs":
        from .gcs_folder import GCSFolder

        logger.debug(f"Creating a GS folder with name: {name}")
        return GCSFolder(
            system=system,
            path=path,
            name=name,
            dryrun=dryrun,
        )

    if system == Folder.DEFAULT_FS:
        logger.debug(f"Creating local folder with name: {name}")
        return Folder(
            system=system,
            path=path,
            name=name,
            dryrun=dryrun,
        )

    from runhouse import Cluster

    cluster_system = _get_cluster_from(system, dryrun=dryrun)
    if isinstance(cluster_system, Cluster):
        logger.debug(f"Creating folder {name} for cluster: {cluster_system.name}")
        return Folder(
            system=cluster_system,
            path=path,
            name=name,
            dryrun=dryrun,
        )

    raise ValueError(
        f"System '{system}' not currently supported. If the file system "
        f"is a cluster (ex: '/my-user/rh-cpu'), make sure the cluster config has been saved."
    )
