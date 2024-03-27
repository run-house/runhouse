import copy
import logging
from pathlib import Path
from typing import Dict, List, Optional

import fsspec

import ray

from runhouse import Folder
from runhouse.globals import rns_client

from runhouse.resources.folders import folder
from runhouse.resources.hardware.utils import _current_cluster, _get_cluster_from
from runhouse.resources.resource import Resource

PREFETCH_KWARG = "prefetch_batches" if ray.__version__ >= "2.4.0" else "prefetch_blocks"

logger = logging.getLogger(__name__)


class Table(Resource):
    RESOURCE_TYPE = "table"
    DEFAULT_FOLDER_PATH = "/runhouse-table"
    DEFAULT_CACHE_FOLDER = ".cache/runhouse/tables"
    DEFAULT_STREAM_FORMAT = "pyarrow"
    DEFAULT_BATCH_SIZE = 256
    DEFAULT_PREFETCH_BATCHES = 1

    def __init__(
        self,
        path: str,
        name: Optional[str] = None,
        file_name: Optional[str] = None,
        system: Optional[str] = None,
        data_config: Optional[dict] = None,
        dryrun: bool = False,
        partition_cols: Optional[List] = None,
        stream_format: Optional[str] = None,
        metadata: Optional[Dict] = None,
        **kwargs,
    ):
        """
        The Runhouse Table object.

        .. note::
            To build a Table, please use the factory method :func:`table`.
        """
        super().__init__(name=name, dryrun=dryrun)
        self.file_name = file_name

        # Use factory method so correct subclass for system is returned
        # strip filename from path if provided
        self._folder = folder(
            path=str(Path(path).parents[0]) if Path(path).suffix else path,
            system=system,
            data_config=data_config,
            dryrun=dryrun,
        )

        self._cached_data = None
        self.partition_cols = partition_cols
        self.stream_format = stream_format or self.DEFAULT_STREAM_FORMAT
        self.metadata = metadata or {}

    @staticmethod
    def from_config(config: dict, dryrun=False):
        if isinstance(config["system"], dict):
            config["system"] = _get_cluster_from(config["system"], dryrun=dryrun)
        return _load_table_subclass(config, dryrun=dryrun)

    def config(self, condensed=True):
        config = super().config(condensed)
        if isinstance(self._folder, Resource):
            config["system"] = self._resource_string_for_subconfig(
                self.system, condensed
            )
            config["data_config"] = self._folder._data_config
        else:
            config["system"] = self.system
        self.save_attrs_to_config(
            config, ["path", "partition_cols", "metadata", "file_name"]
        )
        config.update(config)

        return config

    @property
    def data(self) -> "ray.data.Dataset":
        """Get the table data. If data is not already cached, return a Ray dataset.

        With the dataset object we can stream or convert to other types, for example:

        .. code-block:: python

            data.iter_batches()
            data.to_pandas()
            data.to_dask()
        """
        if self._cached_data is not None:
            return self._cached_data
        return self._read_ray_dataset()

    @data.setter
    def data(self, new_data):
        """Update the data blob to new data"""
        self._cached_data = new_data
        # TODO should we save here?
        # self.save(overwrite=True)

    @property
    def system(self):
        return self._folder.system

    @system.setter
    def system(self, new_system):
        self._folder.system = new_system

    @property
    def path(self):
        if self.file_name:
            return f"{self._folder.path}/{self.file_name}"
        return self._folder.path

    @path.setter
    def path(self, new_path):
        self._folder.path = new_path

    def set_metadata(self, key, val):
        self.metadata[key] = val

    def get_metadata(self, key):
        return self.metadata.get(key)

    @property
    def fsspec_url(self):
        if self.file_name:
            return f"{self._folder.fsspec_url}/{self.file_name}"
        return self._folder.fsspec_url

    @property
    def data_config(self):
        return self._folder.data_config

    @data_config.setter
    def data_config(self, new_data_config):
        self._folder.data_config = new_data_config

    # @classmethod
    # def from_name(cls, name, dryrun=False):
    #     """Load existing Table via its name."""
    #     config = rns_client.load_config(name=name)
    #     if not config:
    #         raise ValueError(f"Table {name} not found.")
    #
    #     # We don't need to load the cluster dict here (if system is a cluster) because the table init
    #     # goes through the Folder factory method, which handles that.
    #
    #     # Add this table's name to the resource artifact registry if part of a run
    #     rns_client.add_upstream_resource(name)
    #
    #     # Uses the table subclass associated with the `resource_subtype`
    #     table_cls = _load_table_subclass(config=config, dryrun=dryrun)
    #     return table_cls.from_config(config=config, dryrun=dryrun)

    def to(self, system, path=None, data_config=None):
        """Copy and return the table on the given filesystem and path.

        Example:
            >>> local_table = rh.table(data, path="local/path")
            >>> s3_table = local_table.to("s3")
            >>> cluster_table = local_table.to(my_cluster)
        """
        new_table = copy.copy(self)
        new_table._folder = self._folder.to(
            system=system, path=path, data_config=data_config
        )
        return new_table

    def _save_sub_resources(self, folder: str = None):
        if isinstance(self.system, Resource):
            self.system.save(folder=folder)

    def write(self):
        """Write underlying table data to fsspec URL.

        Example:
            >>> rh.table(data, path="path/to/write").write()
        """
        import pandas as pd
        import pyarrow as pa
        import ray.data

        if self._cached_data is not None:
            data_to_write = self.data

            if isinstance(data_to_write, pd.DataFrame):
                data_to_write = self._ray_dataset_from_pandas(data_to_write)

            if isinstance(data_to_write, pa.Table):
                data_to_write = self._ray_dataset_from_arrow(data_to_write)

            if not isinstance(data_to_write, ray.data.Dataset):
                raise TypeError(f"Invalid Table format {type(data_to_write)}")

            self._write_ray_dataset(data_to_write)
            logger.info(f"Saved {str(self)} to: {self.fsspec_url}")

        return self

    def fetch(self, columns: Optional[list] = None) -> "pa.Table":
        """Returns the complete table contents.

        Example:
            >>> table = rh.table(data)
            >>> fomratted_data = table.fetch()
        """
        # https://arrow.apache.org/docs/python/generated/pyarrow.parquet.read_table.html
        self._cached_data = self.read_table_from_file(columns)
        if self._cached_data is not None:
            return self._cached_data

        # When trying to read a file like object could lead to IsADirectoryError if the folder path is actually a
        # directory and the file has been automatically generated for us inside the folder
        # (ex: with pyarrow table or with partitioned data that saves multiple files within the directory)

        try:
            import pyarrow.parquet as pq

            table_data = pq.read_table(
                self.path, columns=columns, filesystem=self._folder.fsspec_fs
            )

            if not table_data:
                raise ValueError(f"No table data found in path: {self.path}")

            self._cached_data = table_data
            return self._cached_data

        except:
            raise Exception(f"Failed to read table in path: {self.path}")

    def __getstate__(self):
        """Override the pickle method to clear _cached_data before pickling."""
        state = self.__dict__.copy()
        state["_cached_data"] = None
        return state

    def __iter__(self):
        for block in self.stream(batch_size=self.DEFAULT_BATCH_SIZE):
            for sample in block:
                yield sample

    def __len__(self):
        import pandas as pd
        import ray.data

        if isinstance(self.data, pd.DataFrame):
            len_dataset = self.data.shape[0]

        elif isinstance(self.data, ray.data.Dataset):
            len_dataset = self.data.count()

        else:
            if not hasattr(self.data, "__len__") or not self.data:
                raise RuntimeError("Cannot get len for dataset.")
            else:
                len_dataset = len(self.data)

        return len_dataset

    def __str__(self):
        return self.__class__.__name__

    def stream(
        self,
        batch_size: int,
        drop_last: bool = False,
        shuffle_seed: Optional[int] = None,
        shuffle_buffer_size: Optional[int] = None,
        prefetch_batches: Optional[int] = None,
    ):
        """Return a local batched iterator over the ray dataset.

        Example:
            >>> table = rh.table(data)
            >>> batches = table.stream(batch_size=4)
            >>> for _, batch in batches:
            >>>     print(batch)
        """
        ray_data = self.data

        if self.stream_format == "torch":
            # https://docs.ray.io/en/master/data/api/doc/ray.data.Dataset.iter_torch_batches.html#ray.data.Dataset.iter_torch_batches
            return ray_data.iter_torch_batches(
                batch_size=batch_size,
                drop_last=drop_last,
                local_shuffle_buffer_size=shuffle_buffer_size,
                local_shuffle_seed=shuffle_seed,
                # We need to do this to handle the name change of the prefetch_batches argument in ray 2.4.0
                **{PREFETCH_KWARG: prefetch_batches or self.DEFAULT_PREFETCH_BATCHES},
            )

        elif self.stream_format == "tf":
            # https://docs.ray.io/en/master/data/api/doc/ray.data.Dataset.iter_tf_batches.html
            return ray_data.iter_tf_batches(
                batch_size=batch_size,
                drop_last=drop_last,
                local_shuffle_buffer_size=shuffle_buffer_size,
                local_shuffle_seed=shuffle_seed,
                # We need to do this to handle the name change of the prefetch_batches argument in ray 2.4.0
                **{PREFETCH_KWARG: prefetch_batches or self.DEFAULT_PREFETCH_BATCHES},
            )
        else:
            # https://docs.ray.io/en/latest/data/api/dataset.html#ray.data.Dataset.iter_batches
            return ray_data.iter_batches(
                batch_size=batch_size,
                batch_format=self.stream_format,
                drop_last=drop_last,
                local_shuffle_buffer_size=shuffle_buffer_size,
                local_shuffle_seed=shuffle_seed,
                # We need to do this to handle the name change of the prefetch_batches argument in ray 2.4.0
                **{PREFETCH_KWARG: prefetch_batches or self.DEFAULT_PREFETCH_BATCHES},
            )

    def _read_ray_dataset(self, columns: Optional[List[str]] = None):
        """Read parquet data as a ray dataset object."""
        # https://docs.ray.io/en/latest/data/api/input_output.html#parquet
        dataset = ray.data.read_parquet(
            self.fsspec_url, columns=columns, filesystem=self._folder.fsspec_fs
        )
        return dataset

    def _write_ray_dataset(self, data_to_write: "ray.data.Dataset"):
        """Write a ray dataset to a fsspec filesystem"""
        if self.partition_cols:
            # TODO [JL]: https://arrow.apache.org/docs/python/generated/pyarrow.parquet.write_to_dataset.html
            logger.warning("Partitioning by column not currently supported.")
            pass

        # delete existing contents or they'll just be appended to
        self.rm()

        # https://docs.ray.io/en/master/data/api/doc/ray.data.Dataset.write_parquet.html
        # data_to_write.repartition(os.cpu_count() * 2).write_parquet(
        data_to_write.write_parquet(self.fsspec_url, filesystem=self._folder.fsspec_fs)

    @staticmethod
    def _ray_dataset_from_arrow(data: "pa.Table"):
        """Convert an Arrow Table to a Ray Dataset"""
        import ray.data

        return ray.data.from_arrow(data)

    @staticmethod
    def _ray_dataset_from_pandas(data: "pd.DataFrame"):
        """Convert an Pandas DataFrame to a Ray Dataset"""
        import ray.data

        return ray.data.from_pandas(data)

    def read_table_from_file(self, columns: Optional[list] = None):
        """Read a table from it's path.

        Example:
            >>> table = rh.table(path="path/to/table")
            >>> table_data = table.read_table_from_file()
        """
        try:
            import pyarrow.parquet as pq

            with fsspec.open(self.fsspec_url, mode="rb", **self.data_config) as t:
                table_data = pq.read_table(t.full_name, columns=columns)
            return table_data
        except:
            return None

    def rm(self, recursive: bool = True):
        """Delete table, including its partitioned files where relevant.

        Example:
            >>> table = rh.table(path="path/to/table")
            >>> table.rm()
        """
        # https://filesystem-spec.readthedocs.io/en/latest/api.html#fsspec.spec.AbstractFileSystem.rm
        self._folder.rm(recursive=recursive)

    def exists_in_system(self):
        """Whether the table exists in file system.

        Example:
            >>> table.exists_in_system()
        """
        return (
            self._folder.exists_in_system()
            and len(self._folder.ls(self.fsspec_url)) >= 1
        )


def _load_table_subclass(config: dict, dryrun: bool, data=None):
    """Load the relevant Table subclass based on the config or data type provided"""
    resource_subtype = config.get("resource_subtype", Table.__name__)

    try:
        import datasets

        if resource_subtype == "HuggingFaceTable" or isinstance(data, datasets.Dataset):
            from .huggingface_table import HuggingFaceTable

            return HuggingFaceTable(**config, dryrun=dryrun)
    except ModuleNotFoundError:
        pass
    except Exception as e:
        raise e

    try:
        import pandas as pd

        if resource_subtype == "PandasTable" or isinstance(data, pd.DataFrame):
            from .pandas_table import PandasTable

            return PandasTable(**config, dryrun=dryrun)
    except ModuleNotFoundError:
        pass
    except Exception as e:
        raise e

    try:
        import dask.dataframe as dd

        if resource_subtype == "DaskTable" or isinstance(data, dd.DataFrame):
            from .dask_table import DaskTable

            return DaskTable(**config, dryrun=dryrun)
    except ModuleNotFoundError:
        pass
    except Exception as e:
        raise e

    try:
        import ray.data

        if resource_subtype == "RayTable" or isinstance(data, ray.data.Dataset):
            from .ray_table import RayTable

            return RayTable(**config, dryrun=dryrun)
    except ModuleNotFoundError:
        pass
    except Exception as e:
        raise e

    try:
        import cudf

        if resource_subtype == "CudfTable" or isinstance(data, cudf.DataFrame):
            raise NotImplementedError("Cudf not currently supported")
    except ModuleNotFoundError:
        pass
    except Exception as e:
        raise e

    try:
        import pyarrow as pa

        if resource_subtype == "Table" or isinstance(data, pa.Table):
            new_table = Table(**config, dryrun=dryrun)
            return new_table
    except ModuleNotFoundError:
        pass
    except Exception as e:
        raise e

    raise TypeError(
        f"Unsupported data type {type(data)} for Table construction. "
        f"For converting data to pyarrow see: "
        f"https://arrow.apache.org/docs/7.0/python/generated/pyarrow.Table.html"
    )


def table(
    data=None,
    name: Optional[str] = None,
    path: Optional[str] = None,
    system: Optional[str] = None,
    data_config: Optional[dict] = None,
    partition_cols: Optional[list] = None,
    mkdir: bool = False,
    dryrun: bool = False,
    stream_format: Optional[str] = None,
    metadata: Optional[dict] = None,
) -> Table:
    """Constructs a Table object, which can be used to interact with the table at the given path.

    Args:
        data: Data to be stored in the table.
        name (Optional[str]): Name for the table, to reuse it later on.
        path (Optional[str]): Full path to the data file.
        system (Optional[str]): File system. Currently this must be one of:
            [``file``, ``github``, ``sftp``, ``ssh``, ``s3``, ``gs``, ``azure``].
        data_config (Optional[dict]): The data config to pass to the underlying fsspec handler.
        partition_cols (Optional[list]): List of columns to partition the table by.
        mkdir (bool): Whether to create a remote folder for the table. (Default: ``False``)
        dryrun (bool): Whether to create the Table if it doesn't exist, or load a Table object as a dryrun.
            (Default: ``False``)
        stream_format (Optional[str]): Format to stream the Table as.
            Currently this must be one of: [``pyarrow``, ``torch``, ``tf``, ``pandas``]
        metadata (Optional[dict]): Metadata to store for the table.

    Returns:
        Table: The resulting Table object.

    Example:
        >>> import runhouse as rh
        >>> # Create and save (pandas) table
        >>> rh.table(
        >>>    data=data,
        >>>    name="~/my_test_pandas_table",
        >>>    path="table_tests/test_pandas_table.parquet",
        >>>    system="file",
        >>>    mkdir=True,
        >>> ).save()
        >>>
        >>> # Load table from above
        >>> reloaded_table = rh.table(name="~/my_test_pandas_table")
    """
    if name and not any(
        [
            data is not None,
            path,
            system,
            data_config,
            partition_cols,
            stream_format,
            metadata,
        ]
    ):
        # Try reloading existing table
        return Table.from_name(name, dryrun)

    system = _get_cluster_from(
        system or _current_cluster(key="config") or Folder.DEFAULT_FS, dryrun=dryrun
    )

    file_name = None
    if path:
        # Extract the file name from the path if provided
        full_path = Path(path)
        file_suffix = full_path.suffix
        if file_suffix:
            path = str(full_path.parent)
            file_name = full_path.name

    if path is None:
        # If no path is provided we need to create one based on the name of the table
        table_name_in_path = rns_client.resolve_rns_data_resource_name(name)
        if system == rns_client.DEFAULT_FS or (
            isinstance(system, Resource) and system.on_this_cluster()
        ):
            # create random path to store in .cache folder of local filesystem
            path = str(
                Path(
                    f"~/{Table.DEFAULT_CACHE_FOLDER}/{table_name_in_path}"
                ).expanduser()
            )
        else:
            # save to the default bucket
            path = f"{Table.DEFAULT_FOLDER_PATH}/{table_name_in_path}"

    config = {
        "system": system,
        "name": name,
        "path": path,
        "file_name": file_name,
        "data_config": data_config,
        "partition_cols": partition_cols,
        "stream_format": stream_format,
        "metadata": metadata,
    }

    new_table = _load_table_subclass(config=config, dryrun=dryrun, data=data)
    if data is not None:
        new_table.data = data

    return new_table
