import copy
import logging
from pathlib import Path
from typing import Dict, Optional

import runhouse as rh
from runhouse.rh_config import rns_client
from runhouse.rns.api_utils.utils import generate_uuid
from runhouse.rns.folders.folder import Folder, folder
from runhouse.rns.obj_store import _current_cluster
from runhouse.rns.resource import Resource

logger = logging.getLogger(__name__)


class Blob(Resource):
    RESOURCE_TYPE = "blob"
    DEFAULT_FOLDER_PATH = "/runhouse-blob"
    DEFAULT_CACHE_FOLDER = ".cache/runhouse/blobs"

    def __init__(
        self,
        path: Optional[str] = None,
        name: Optional[str] = None,
        system: Optional[str] = Folder.DEFAULT_FS,
        data_config: Optional[Dict] = None,
        dryrun: bool = False,
        **kwargs,
    ):
        """
        Runhouse Blob object

        .. note::
                To build a Blob, please use the factory method :func:`blob`.
        """
        super().__init__(name=name, dryrun=dryrun)
        self._filename = str(Path(path).name) if path else self.name
        # Use factory method so correct subclass for system is returned
        self._folder = folder(
            path=str(Path(path).parent) if path is not None else path,
            system=system,
            data_config=data_config,
            dryrun=dryrun,
        )
        self._cached_data = None

    @property
    def config_for_rns(self):
        config = super().config_for_rns
        blob_config = {
            "path": self.path,  # pair with data source to create the physical URL
            "resource_type": self.RESOURCE_TYPE,
            "system": self.system,
        }
        config.update(blob_config)
        return config

    @staticmethod
    def from_config(config: dict, dryrun=True):
        return Blob(**config, dryrun=dryrun)

    @property
    def data(self):
        """Get the blob data."""
        if self._cached_data is not None:
            return self._cached_data
        data = self.fetch()
        return data

    @data.setter
    def data(self, new_data):
        """Update the data blob to new data."""
        self._cached_data = new_data

    @property
    def system(self):
        return self._folder.system

    @system.setter
    def system(self, new_system):
        self._folder.system = new_system

    @property
    def path(self):
        return self._folder.path + "/" + self._filename

    @path.setter
    def path(self, new_path):
        self._folder.path = str(Path(new_path).parent)
        self._filename = str(Path(new_path).name)

    @property
    def data_config(self):
        return self._folder.data_config

    @data_config.setter
    def data_config(self, new_data_config):
        self._folder.data_config = new_data_config

    @property
    def fsspec_url(self):
        return self._folder.fsspec_url + "/" + self._filename

    def open(self, mode: str = "rb"):
        """Get a file-like (OpenFile container object) of the blob data. User must close the file, or use this
        method inside of a with statement (e.g. `with my_blob.open() as f:`)."""
        return self._folder.open(self._filename, mode=mode)

    def to(
        self, system, path: Optional[str] = None, data_config: Optional[dict] = None
    ):
        """Return a copy of the blob on the destination system and path."""
        new_blob = copy.copy(self)
        new_blob._folder = self._folder.to(
            system=system, path=path, data_config=data_config
        )
        return new_blob

    def fetch(self):
        """Return the data for the user to deserialize"""
        self._cached_data = self._folder.get(self._filename)
        return self._cached_data

    def _save_sub_resources(self):
        if isinstance(self.system, Resource):
            self.system.save()

    def write(self):
        """Save the underlying blob to its specified fsspec URL."""
        # TODO figure out default behavior for not overwriting but still saving
        # if not overwrite:
        #     TODO check if data_url is already in use
        #     time = datetime.today().strftime('%Y-%m-%d_%H:%M:%S')
        #     self.data_url = self.data_url + time or time

        # TODO check if self._cached_data is None, and if so, don't just download it to then save it again?
        with self.open(mode="wb") as f:
            if not isinstance(self.data, bytes):
                # Avoid TypeError: a bytes-like object is required
                raise TypeError(
                    f"Cannot save blob with data of type {type(self.data)}, data must be serialized"
                )

            f.write(self.data)

        return self

    def delete_in_system(self):
        """Delete the blob and the folder it lives in from the file system."""
        self._folder.rm(self._filename)
        if self.system == "file":
            # Deleting the blob itself in a local file system will not remove the parent folder by default
            self._folder.delete_in_system()

    def exists_in_system(self):
        """Check whether the blob exists in the file system"""
        return self._folder.fsspec_fs.exists(self.fsspec_url)

    # TODO [DG] get rid of this in favor of just "sync_down(path, system)" ?
    def sync_from_cluster(self, cluster, path: Optional[str] = None):
        """Efficiently rsync down a blob from a cluster, into the path of the current Blob object."""
        if not cluster.address:
            raise ValueError("Cluster must be started before copying data to it.")

        cluster.rsync(source=self.path, dest=path, up=False)


def blob(
    data=None,
    name: Optional[str] = None,
    path: Optional[str] = None,
    system: Optional[str] = None,
    data_config: Optional[Dict] = None,
    mkdir: bool = False,
    dryrun: bool = False,
    load: bool = True,
):
    """Returns a Blob object, which can be used to interact with the resource at the given path

    Args:
        data: Blob data. This should be provided as a serialized object.
        name (Optional[str]): Name to give the blob object, to be reused later on.
        path (Optional[str]): Path (or path) of the blob object.
        system (Optional[str]): File system. Currently this must be one of:
           [``file``, ``github``, ``sftp``, ``ssh``,``s3``, ``gs``, ``azure``].
            We are working to add additional file system support.
        data_config (Optional[Dict]): The data config to pass to the underlying fsspec handler.
        mkdir (bool): Whether to create a remote folder for the blob. (Default: ``False``)
        dryrun (bool): Whether to create the Blob if it doesn't exist, or load a Blob object as a dryrun.
            (Default: ``False``)
        load (bool): Whether to load an existing config for the Blob. (Default: ``True``)

    Returns:
        Blob: The resulting blob.

    Example:
        >>> data = json.dumps(list(range(50))
        >>>
        >>> # Remote blob with name and no path (saved to bucket called runhouse/blobs/my-blob)
        >>> rh.blob(name="@/my-blob", data=data, data_source='s3', dryrun=False)
        >>>
        >>> # Remote blob with name and path
        >>> rh.blob(name='@/my-blob', path='/runhouse-tests/my_blob.pickle', data=data, system='s3', dryrun=False)
        >>>
        >>> # Local blob with name and path, save to local filesystem
        >>> rh.blob(name=name, data=data, path=str(Path.cwd() / "my_blob.pickle"), dryrun=False)
        >>>
        >>> # Local blob with name and no path (saved to ~/.cache/blobs/my-blob)
        >>> rh.blob(name="~/my-blob", data=data, dryrun=False)

        >>> # Loading a blob
        >>> my_local_blob = rh.blob(name="~/my_blob")
        >>> my_s3_blob = rh.blob(name="@/my_blob")
    """
    config = rns_client.load_config(name) if load else {}

    system = (
        system
        or config.get("system")
        or _current_cluster(key="config")
        or Folder.DEFAULT_FS
    )
    config["system"] = system

    name = name or config.get("rns_address") or config.get("name")

    data_path = path or config.get("path")
    if data_path is None:
        blob_name_in_path = (
            f"{generate_uuid()}/{rns_client.resolve_rns_data_resource_name(name)}"
        )

        if (
            system == rns_client.DEFAULT_FS
            or config["system"] == _current_cluster()
            or (
                isinstance(config["system"], dict)
                and config["system"]["name"] == _current_cluster()
            )
        ):
            # create random path to store in .cache folder of local filesystem
            data_path = str(
                Path(f"~/{Blob.DEFAULT_CACHE_FOLDER}/{blob_name_in_path}").expanduser()
            )
        else:
            # save to the default bucket
            data_path = f"{Blob.DEFAULT_FOLDER_PATH}/{blob_name_in_path}"

    config["name"] = name
    config["path"] = data_path
    config["data_config"] = data_config or config.get("data_config")

    if isinstance(config["system"], str) and rns_client.exists(
        config["system"], resource_type="cluster"
    ):
        config["system"] = rns_client.load_config(config["system"])
    elif isinstance(config["system"], dict):
        from runhouse.rns.hardware.cluster import Cluster

        config["system"] = Cluster.from_config(config["system"])

    if mkdir:
        # create the remote folder for the blob
        folder_path = str(Path(data_path).parent)
        rh.folder(name=folder_path, system=system, dryrun=True).mkdir()

    new_blob = Blob.from_config(config, dryrun=dryrun)
    new_blob.data = data

    return new_blob
