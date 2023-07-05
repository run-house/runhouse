import logging
from typing import Any, Dict, Optional, Union

from runhouse.rh_config import obj_store
from runhouse.rns.hardware.cluster import Cluster
from runhouse.rns.resource import Resource
from runhouse.rns.utils.hardware import _current_cluster, _get_cluster_from
from runhouse.rns.utils.names import _generate_default_name, _generate_default_path

logger = logging.getLogger(__name__)


class Blob(Resource):
    RESOURCE_TYPE = "blob"
    DEFAULT_FOLDER_PATH = "/runhouse-blob"
    DEFAULT_CACHE_FOLDER = ".cache/runhouse/blobs"

    def __init__(
        self,
        name: Optional[str] = None,
        system: Union[None, str, Cluster] = None,
        dryrun: bool = False,
        **kwargs,
    ):
        """
        Runhouse Blob object

        .. note::
                To build a Blob, please use the factory method :func:`blob`.
        """
        super().__init__(name=name, dryrun=dryrun)
        self._system = system
        # _data is only used when not running on a cluster (i.e. no obj store)
        self._data = None

    @property
    def config_for_rns(self):
        if not self.system:
            raise ValueError(
                "Cannot save an in-memory local blob to RNS. Please send the blob to a local "
                "path or system first."
            )
        config = super().config_for_rns
        blob_config = {
            "system": self._resource_string_for_subconfig(self.system)
            if self.system
            else None,
        }
        config.update(blob_config)
        return config

    @staticmethod
    def from_config(config: dict, dryrun=False):
        if config["resource_subtype"] == "File":
            from runhouse.rns.blobs.file import File

            return File(**config, dryrun=dryrun)
        return Blob(**config, dryrun=dryrun)

    @classmethod
    def _check_for_child_configs(cls, config):
        """Overload by child resources to load any resources they hold internally."""
        system = config["system"]
        if isinstance(system, str):
            config["system"] = _get_cluster_from(system)
        return config

    @property
    def system(self):
        return self._system

    @system.setter
    def system(self, new_system):
        self._system = _get_cluster_from(new_system)

    def to(
        self,
        system: Union[str, Cluster],
        path: Optional[str] = None,
        data_config: Optional[dict] = None,
    ):
        """Return a copy of the blob on the destination system, and optionally path.

        Example:
            >>> local_blob = rh.blob(data)
            >>> s3_blob = blob.to("s3")
            >>> cluster_blob = blob.to(my_cluster)
        """
        if system == "here":
            if not path:
                current_cluster_config = _current_cluster(key="config")
                if current_cluster_config:
                    system = Cluster.from_config(current_cluster_config)
                else:
                    system = None
            else:
                system = "file"

        system = _get_cluster_from(system)
        if (not system or isinstance(system, Cluster)) and not path:
            name = self.name or _generate_default_name(prefix="blob")
            return Blob(name=name, system=system).write(self.fetch())

        path = str(
            path or self.default_path(self.rns_address, system)
        )  # Make sure it's a string and not a Path

        from runhouse.rns.blobs.file import File

        new_blob = File(path=path, system=system, data_config=data_config)
        new_blob.write(self.fetch())
        return new_blob

    def fetch(self):
        """Return the data for the user to deserialize.

        Example:
            >>> data = blob.fetch()
        """
        if not self.system:
            return self._data
        if self.system.on_this_cluster():
            return obj_store.get(self.name)
        return self.system.get(self.name, stream_logs=False)

    def _save_sub_resources(self):
        if isinstance(self.system, Resource):
            self.system.save()

    def rename(self, name: str):
        """Rename the blob.

        Example:
            >>> blob = rh.blob(data)
            >>> blob.rename("new_name")
        """
        if self.name == name:
            return
        old_name = self.name
        self.name = name  # Goes through Resource setter to parse name properly (e.g. if rns path)
        # Also check that this is a Blob and not a File
        if isinstance(self.system, Cluster) and self.__class__.__name__ == "Blob":
            if self.system.on_this_cluster():
                obj_store.rename(old_key=old_name, new_key=self.name)
            else:
                self.system.rename(old_key=old_name, new_key=self.name)

    def save(
        self,
        name: str = None,
        overwrite: bool = True,
    ):
        # Need to override Resource's save to handle key changes in the obj store
        # Also check that this is a Blob and not a File
        if name and not self.name == name and self.__class__.__name__ == "Blob":
            if overwrite:
                self.rename(name)
            else:
                if isinstance(self.system, Cluster):
                    if self.system.on_this_cluster():
                        obj_store.put(self.name, self.fetch())
                    else:
                        self.system.put(name, self.fetch())
        super().save(name=name, overwrite=overwrite)

    def write(self, data):
        """Save the underlying blob to its cluster's store.

        Example:
            >>> rh.blob(data).write()
        """
        if not self.system:
            self._data = data
        elif self.system.on_this_cluster():
            obj_store.put(self.name, data)
        else:
            self.system.put(self.name, data)
        return self

    def rm(self):
        """Delete the blob from wherever it's stored.

        Example:
            >>> blob = rh.blob(data)
            >>> blob.rm()
        """
        if self.system is None:
            self._data = None
        if self.system.on_this_cluster():
            obj_store.delete(self.name)
        else:
            self.system.delete(self.name)

    def exists_in_system(self):
        """Check whether the blob exists in the file system

        Example:
            >>> blob = rh.blob(data)
            >>> blob.exists_in_system()
        """
        if self.system is None:
            return True
        if self.system.on_this_cluster():
            return obj_store.exists(self.name)
        return self.name in self.system.list_keys()


def blob(
    data: [Any] = None,
    name: Optional[str] = None,
    path: Optional[str] = None,
    system: Optional[str] = None,
    data_config: Optional[Dict] = None,
    dryrun: bool = False,
):
    """Returns a Blob object, which can be used to interact with the resource at the given path

    Args:
        data: Blob data. The data to persist either on the cluster or in the filesystem.
        name (Optional[str]): Name to give the blob object, to be reused later on.
        path (Optional[str]): Path (or path) to the blob object. Specfying a path will force the blob to be
            saved to the filesystem rather than persist in the cluster's object store.
        system (Optional[str or Cluster]): File system or cluster name. If providing a file system this must be one of:
            [``file``, ``github``, ``sftp``, ``ssh``, ``s3``, ``gs``, ``azure``].
            We are working to add additional file system support. If providing a cluster, this must be a cluster object
            or name, and whether the data is saved to the object store or filesystem depends on whether a path is
            specified.
        data_config (Optional[Dict]): The data config to pass to the underlying fsspec handler (in the case of
            saving the the filesystem).
        dryrun (bool): Whether to create the Blob if it doesn't exist, or load a Blob object as a dryrun.
            (Default: ``False``)

    Returns:
        Blob: The resulting blob.

    Example:
        >>> import runhouse as rh
        >>> import json
        >>>
        >>> data = list(range(50)
        >>> serialized_data = json.dumps(data)
        >>>
        >>> # Local blob with name and no path (saved to Runhouse object store)
        >>> rh.blob(name="@/my-blob", data=data)
        >>>
        >>> # Remote blob with name and no path (saved to cluster's Runhouse object store)
        >>> rh.blob(name="@/my-blob", data=data, system=my_cluster)
        >>>
        >>> # Remote blob with name, filesystem, and no path (saved to filesystem with default path)
        >>> rh.blob(name="@/my-blob", data=serialized_data, system="s3")
        >>>
        >>> # Remote blob with name and path (saved to remote filesystem)
        >>> rh.blob(name='@/my-blob', data=serialized_data, path='/runhouse-tests/my_blob.pickle', system='s3')
        >>>
        >>> # Local blob with path and no system (saved to local filesystem)
        >>> rh.blob(data=serialized_data, path=str(Path.cwd() / "my_blob.pickle"))

        >>> # Loading a blob
        >>> my_local_blob = rh.blob(name="~/my_blob")
        >>> my_s3_blob = rh.blob(name="@/my_blob")
    """
    if name and not any([data is not None, path, system, data_config]):
        # Try reloading existing blob
        try:
            return Blob.from_name(name, dryrun)
        except ValueError:
            # This is a rare instance where passing no constructor params is actually valid
            # (e.g. rh.blob(name=key).fetch()), so if we don't find the name, we still want to
            # create a new blob.
            pass

    system = _get_cluster_from(system or _current_cluster(key="config"), dryrun=dryrun)

    if (not system or isinstance(system, Cluster)) and not path and data_config is None:
        # Blobs must be named, or we don't have a key for the kv store
        name = name or _generate_default_name(prefix="blob")
        new_blob = Blob(name=name, system=system, dryrun=dryrun)
        if data is not None:
            new_blob.write(data)
        return new_blob

    path = str(path or _generate_default_path(Blob, name, system))

    from runhouse.rns.blobs.file import File

    new_blob = File(
        name=name, path=path, system=system, data_config=data_config, dryrun=dryrun
    )
    if data is not None:
        new_blob.write(data)
    return new_blob
