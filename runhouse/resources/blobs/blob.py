import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union

from runhouse.resources.envs.env import Env
from runhouse.resources.envs.utils import _get_env_from
from runhouse.resources.hardware import _current_cluster, _get_cluster_from, Cluster

from runhouse.resources.module import Module
from runhouse.rns.utils.names import _generate_default_name, _generate_default_path

logger = logging.getLogger(__name__)


class Blob(Module):
    RESOURCE_TYPE = "blob"
    DEFAULT_FOLDER_PATH = "/runhouse-blob"
    DEFAULT_CACHE_FOLDER = ".cache/runhouse/blobs"

    def __init__(
        self,
        name: Optional[str] = None,
        system: Union[Cluster, str] = None,
        env: Optional[Env] = None,
        dryrun: bool = False,
        **kwargs,
    ):
        """
        Runhouse Blob object

        .. note::
                To build a Blob, please use the factory method :func:`blob`.
        """
        self.data = None
        super().__init__(name=name, system=system, env=env, dryrun=dryrun, **kwargs)

    def to(
        self,
        system: Union[str, Cluster],
        env: Optional[Union[str, Env]] = None,
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
            self.name = self.name or _generate_default_name(prefix="blob")
            # TODO [DG] if system is the same, bounces off the laptop for no reason. Change to write through a
            #  call_module_method rpc (and same for similar file cases)
            return super().to(system, env)

        from runhouse import Folder

        path = str(
            path or Folder.default_path(self.rns_address, system)
        )  # Make sure it's a string and not a Path

        from runhouse.resources.blobs.file import file

        new_blob = file(path=path, system=system, data_config=data_config)
        new_blob.write(self.fetch())
        return new_blob

    # TODO delete
    def write(self, data):
        """Save the underlying blob to its cluster's store.

        Example:
            >>> rh.blob(data).write()
        """
        self.data = data

    def rm(self):
        """Delete the blob from wherever it's stored.

        Example:
            >>> blob = rh.blob(data)
            >>> blob.rm()
        """
        self.data = None

    def exists_in_system(self):
        """Check whether the blob exists in the file system

        Example:
            >>> blob = rh.blob(data)
            >>> blob.exists_in_system()
        """
        if self.data is not None:
            return True

    def resolved_state(self, _state_dict=None):
        """Return the resolved state of the blob, which is the data.

        Primarily used to define the behavior of the ``fetch`` method.

        Example:
            >>> blob = rh.blob(data)
            >>> blob.resolved_state()
        """
        return self.data


def blob(
    data: [Any] = None,
    name: Optional[str] = None,
    path: Optional[Union[str, Path]] = None,
    system: Optional[str] = None,
    env: Optional[Union[str, Env]] = None,
    data_config: Optional[Dict] = None,
    load: bool = True,
    dryrun: bool = False,
):
    """Returns a Blob object, which can be used to interact with the resource at the given path

    Args:
        data: Blob data. The data to persist either on the cluster or in the filesystem.
        name (Optional[str]): Name to give the blob object, to be reused later on.
        path (Optional[str or Path]): Path (or path) to the blob object. Specfying a path will force the blob to be
            saved to the filesystem rather than persist in the cluster's object store.
        system (Optional[str or Cluster]): File system or cluster name. If providing a file system this must be one of:
            [``file``, ``github``, ``sftp``, ``ssh``, ``s3``, ``gs``, ``azure``].
            We are working to add additional file system support. If providing a cluster, this must be a cluster object
            or name, and whether the data is saved to the object store or filesystem depends on whether a path is
            specified.
        env (Optional[Env or str]): Environment for the blob. If left empty, defaults to base environment.
            (Default: ``None``)
        data_config (Optional[Dict]): The data config to pass to the underlying fsspec handler (in the case of
            saving the the filesystem).
        load (bool): Whether to try to load the Blob object from RNS. (Default: ``True``)
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
    if name and load and not any([data is not None, path, system, data_config]):
        # Try reloading existing blob
        try:
            return Blob.from_name(name, dryrun)
        except ValueError:
            # This is a rare instance where passing no constructor params is actually valid
            # (e.g. rh.blob(name=key).write(data)), so if we don't find the name, we still want to
            # create a new blob.
            pass

    system = _get_cluster_from(system or _current_cluster(key="config"), dryrun=dryrun)
    env = env or _get_env_from(env)

    if (not system or isinstance(system, Cluster)) and not path and data_config is None:
        # Blobs must be named, or we don't have a key for the kv store
        name = name or _generate_default_name(prefix="blob")
        new_blob = Blob(name=name, dryrun=dryrun).to(system, env)
        if data is not None:
            new_blob.data = data
        return new_blob

    path = str(path or _generate_default_path(Blob, name, system))

    from runhouse.resources.blobs.file import File

    name = name or _generate_default_name(prefix="file")
    new_blob = File(
        name=name,
        path=path,
        system=system,
        env=env,
        data_config=data_config,
        dryrun=dryrun,
    )
    if isinstance(system, Cluster):
        system.put_resource(new_blob)
    if data is not None:
        new_blob.write(data)
    return new_blob
