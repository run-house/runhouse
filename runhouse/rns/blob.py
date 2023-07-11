import copy
import logging
from pathlib import Path
from typing import Dict, Optional

from runhouse.rh_config import rns_client
from runhouse.rns.api_utils.utils import generate_uuid
from runhouse.rns.folders import Folder, folder
from runhouse.rns.resource import Resource
from runhouse.rns.utils.hardware import _current_cluster, _get_cluster_from

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
            "system": self._resource_string_for_subconfig(self.system),
        }
        config.update(blob_config)
        return config

    @staticmethod
    def from_config(config: dict, dryrun=False):
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
        """Get a file-like (OpenFile container object) of the blob data.
        User must close the file, or use this method inside of a with statement.

        Example:
            >>> with my_blob.open(mode="wb") as f:
            >>>     f.write(data)
            >>>
            >>> obj = my_blob.open()
        """
        return self._folder.open(self._filename, mode=mode)

    def to(
        self, system, path: Optional[str] = None, data_config: Optional[dict] = None
    ):
        """Return a copy of the blob on the destination system and path.

        Example:
            >>> local_blob = rh.blob(data)
            >>> s3_blob = blob.to("s3")
            >>> cluster_blob = blob.to(my_cluster)
        """
        new_blob = copy.copy(self)
        new_blob._folder = self._folder.to(
            system=system, path=path, data_config=data_config
        )
        return new_blob

    def fetch(self):
        """Return the data for the user to deserialize.

        Example:
            >>> serialized_data = blob.fetch()
        """
        self._cached_data = self._folder.get(self._filename)
        return self._cached_data

    def _save_sub_resources(self):
        if isinstance(self.system, Resource):
            self.system.save()

    def write(self):
        """Save the underlying blob to its specified fsspec URL.

        Example:
            >>> rh.blob(serialized_data, path="path/to/save").write()
        """
        # TODO figure out default behavior for not overwriting but still saving
        # if not overwrite:
        #     TODO check if data_url is already in use
        #     time = datetime.today().strftime('%Y-%m-%d_%H:%M:%S')
        #     self.data_url = self.data_url + time or time

        # TODO check if self._cached_data is None, and if so, don't just download it to then save it again?
        self._folder.mkdir()
        with self.open(mode="wb") as f:
            if not isinstance(self.data, bytes):
                # Avoid TypeError: a bytes-like object is required
                raise TypeError(
                    f"Cannot save blob with data of type {type(self.data)}, data must be serialized"
                )

            f.write(self.data)

        return self

    def rm(self):
        """Delete the blob and the folder it lives in from the file system.

        Example:
            >>> blob = rh.blob(serialized_data, path="saved/path")
            >>> blob.rm()
        """
        self._folder.rm(contents=[self._filename], recursive=False)

    def exists_in_system(self):
        """Check whether the blob exists in the file system

        Example:
            >>> blob = rh.blob(serialized_data, path="saved/path")
            >>> blob.exists_in_system()
        """
        return self._folder.fsspec_fs.exists(self.fsspec_url)

    # TODO [DG] get rid of this in favor of just "sync_down(path, system)" ?
    def sync_from_cluster(self, cluster, path: Optional[str] = None):
        """Efficiently rsync down a blob from a cluster, into the path of the current Blob object.

        Example:
            >>> remote_blob = rh.blob(serialized_data, system=my_cluster)
            >>> remote_blob.sync_from_cluster()
        """
        if not cluster.address:
            raise ValueError("Cluster must be started before copying data to it.")

        cluster._rsync(source=self.path, dest=path, up=False)


def blob(
    data=None,
    name: Optional[str] = None,
    path: Optional[str] = None,
    system: Optional[str] = None,
    data_config: Optional[Dict] = None,
    dryrun: bool = False,
):
    """Returns a Blob object, which can be used to interact with the resource at the given path

    Args:
        data: Blob data. This should be provided as a serialized object.
        name (Optional[str]): Name to give the blob object, to be reused later on.
        path (Optional[str]): Path (or path) of the blob object.
        system (Optional[str or Cluster]): File system or cluster name. If providing a file system this must be one of:
            [``file``, ``github``, ``sftp``, ``ssh``, ``s3``, ``gs``, ``azure``].
            We are working to add additional file system support.
        data_config (Optional[Dict]): The data config to pass to the underlying fsspec handler.
        dryrun (bool): Whether to create the Blob if it doesn't exist, or load a Blob object as a dryrun.
            (Default: ``False``)

    Returns:
        Blob: The resulting blob.

    Example:
        >>> import runhouse as rh
        >>> import json
        >>> data = json.dumps(list(range(50))
        >>>
        >>> # Remote blob with name and no path (saved to bucket called runhouse/blobs/my-blob)
        >>> rh.blob(name="@/my-blob", data=data, system='s3').write()
        >>>
        >>> # Remote blob with name and path
        >>> rh.blob(name='@/my-blob', path='/runhouse-tests/my_blob.pickle', system='s3').save()
        >>>
        >>> # Local blob with name and path, save to local filesystem
        >>> rh.blob(data=data, path=str(Path.cwd() / "my_blob.pickle")).write()
        >>>
        >>> # Local blob with name and no path (saved to ~/.cache/blobs/my-blob)
        >>> rh.blob(name="~/my-blob", data=data).write().save()

        >>> # Loading a blob
        >>> my_local_blob = rh.blob(name="~/my_blob")
        >>> my_s3_blob = rh.blob(name="@/my_blob")
    """
    if name and not any([data, path, system, data_config]):
        # Try reloading existing blob
        return Blob.from_name(name, dryrun)

    system = _get_cluster_from(
        system or _current_cluster(key="config") or Folder.DEFAULT_FS, dryrun=dryrun
    )

    if path is None:
        blob_name_in_path = (
            f"{generate_uuid()}/{rns_client.resolve_rns_data_resource_name(name)}"
        )

        if system == rns_client.DEFAULT_FS or (
            isinstance(system, Resource) and system.on_this_cluster()
        ):
            # create random path to store in .cache folder of local filesystem
            path = str(
                Path(f"~/{Blob.DEFAULT_CACHE_FOLDER}/{blob_name_in_path}").expanduser()
            )
        else:
            # save to the default bucket
            path = f"{Blob.DEFAULT_FOLDER_PATH}/{blob_name_in_path}"

    new_blob = Blob(
        path=path, system=system, data_config=data_config, name=name, dryrun=dryrun
    )
    new_blob.data = data

    return new_blob
