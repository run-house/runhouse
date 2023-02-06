import copy
import logging
import uuid
from pathlib import Path
from typing import Dict, List, Optional

import fsspec
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import ray.data

import runhouse as rh
from runhouse.rh_config import rns_client
from runhouse.rns.folders.folder import folder
from .. import Resource, SkyCluster
from ..top_level_rns_fns import save

logger = logging.getLogger(__name__)


class Table(Resource):
    RESOURCE_TYPE = "table"
    DEFAULT_FOLDER_PATH = "/runhouse/tables"
    DEFAULT_CACHE_FOLDER = ".cache/runhouse/tables"
    DEFAULT_STREAM_FORMAT = "pyarrow"
    DEFAULT_BATCH_SIZE = 256
    DEFAULT_PREFETCH_BLOCKS = 1

    def __init__(
        self,
        url: str,
        name: Optional[str] = None,
        file_name: Optional[str] = None,
        fs: Optional[str] = None,
        data_config: Optional[dict] = None,
        dryrun: bool = True,
        partition_cols: Optional[List] = None,
        stream_format: Optional[str] = None,
        metadata: Optional[Dict] = None,
        **kwargs,
    ):
        super().__init__(name=name, dryrun=dryrun)
        self._filename = str(Path(url).name) if url else self.name
        # Use factory method so correct subclass for fs is returned
        self._folder = folder(url=url, fs=fs, data_config=data_config, dryrun=dryrun)
        self._cached_data = None
        self.partition_cols = partition_cols
        self.file_name = file_name
        self.stream_format = stream_format or self.DEFAULT_STREAM_FORMAT
        self.metadata = metadata or {}

    @staticmethod
    def from_config(config: dict, dryrun=True):
        if isinstance(config["fs"], dict):
            config["fs"] = SkyCluster.from_config(config["fs"], dryrun=dryrun)
        return Table(**config, dryrun=dryrun)

    @property
    def config_for_rns(self):
        config = super().config_for_rns
        if isinstance(self._folder, Resource):
            config["fs"] = self._resource_string_for_subconfig(self.fs)
            config["data_config"] = self._folder._data_config
        else:
            config["fs"] = self.fs
        self.save_attrs_to_config(config, ["url", "partition_cols", "metadata"])
        config.update(config)

        return config

    @property
    def data(self) -> ray.data.Dataset:
        """Get the table data. If data is not already cached return a ray dataset. With the dataset object we can
        stream or convert to other types, for example:
            data.iter_batches()
            data.to_pandas()
            data.to_dask()
        """
        if self._cached_data is not None:
            return self._cached_data
        return self.read_ray_dataset()

    @data.setter
    def data(self, new_data):
        """Update the data blob to new data"""
        self._cached_data = new_data
        # TODO should we save here?
        # self.save(overwrite=True)

    @property
    def fs(self):
        return self._folder.fs

    @fs.setter
    def fs(self, new_fs):
        self._folder.fs = new_fs

    @property
    def url(self):
        if self.file_name:
            return f"{self._folder.url}/{self.file_name}"
        return self._folder.url

    @url.setter
    def url(self, new_url):
        self._folder.url = new_url

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

    def to(self, fs, url=None, data_config=None):
        new_table = copy.copy(self)
        new_table._folder = self._folder.to(fs=fs, url=url, data_config=data_config)
        return new_table

    def save(
        self,
        name: Optional[str] = None,
        snapshot: bool = False,
        overwrite: bool = True,
        **snapshot_kwargs,
    ):
        if self._cached_data is not None:
            data_to_write = self.data
            if isinstance(data_to_write, pa.Table):
                data_to_write = self.ray_dataset_from_arrow(data_to_write)

            if not isinstance(data_to_write, ray.data.Dataset):
                raise TypeError(f"Invalid Table format {type(data_to_write)}")

            self.write_ray_dataset(data_to_write)
            logger.info(f"Saved {str(self)} to: {self.fsspec_url}")

        save(self, snapshot=snapshot, overwrite=overwrite, **snapshot_kwargs)

        return self

    def fetch(self, columns: Optional[list] = None) -> pa.Table:
        # https://arrow.apache.org/docs/python/generated/pyarrow.parquet.read_table.html
        try:
            with fsspec.open(self.fsspec_url, mode="rb", **self.data_config) as t:
                self._cached_data = pq.read_table(t.full_name, columns=columns)
        except:
            # When trying to read as file like object could fail for a couple of reasons:
            # IsADirectoryError: The folder URL is actually a directory and the file has been automatically
            # generated for us inside the folder (ex: pyarrow table)

            # The file system is SFTP: since the SFTPFileSystem takes the host as a separate param, we cannot
            # pass in the data config as a single data_config kwarg

            # When specifying the filesystem don't pass in the fsspec url (which includes the file system prepended)
            self._cached_data = pq.read_table(
                self.url, columns=columns, filesystem=self._folder.fsspec_fs
            )
        return self._cached_data

    def __getstate__(self):
        """Override the pickle method to clear _cached_data before pickling"""
        state = self.__dict__.copy()
        state["_cached_data"] = None
        return state

    def __iter__(self):
        for block in self.stream(batch_size=self.DEFAULT_BATCH_SIZE):
            for sample in block:
                yield sample

    def __len__(self):
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
        prefetch_blocks: Optional[int] = None,
    ):
        """Return a local batched iterator over the ray dataset.

        Args:
            batch_size: The number of rows in each batch. The final batch may include fewer than ``batch_size`` rows if
                ``drop_last`` is ``False``.
            drop_last: Whether to drop the last batch if it's incomplete.
            shuffle_seed: The seed to use for the local random shuffle.
            shuffle_buffer_size: minimum number of rows that must be in the local in-memory shuffle
                buffer in order to yield a batch. Must be greater than or equal to ``batch_size``.
            prefetch_blocks: The number of blocks to prefetch ahead of the current block during the scan.
        """
        ray_data = self.data

        if self.stream_format == "torch":
            # https://docs.ray.io/en/master/data/api/doc/ray.data.Dataset.iter_torch_batches.html#ray.data.Dataset.iter_torch_batches
            return ray_data.iter_torch_batches(
                batch_size=batch_size,
                prefetch_blocks=prefetch_blocks or self.DEFAULT_PREFETCH_BLOCKS,
                drop_last=drop_last,
                local_shuffle_buffer_size=shuffle_buffer_size,
                local_shuffle_seed=shuffle_seed,
            )

        elif self.stream_format == "tf":
            # https://docs.ray.io/en/master/data/api/doc/ray.data.Dataset.iter_tf_batches.html
            return ray_data.iter_tf_batches(
                batch_size=batch_size,
                prefetch_blocks=prefetch_blocks or self.DEFAULT_PREFETCH_BLOCKS,
                drop_last=drop_last,
                local_shuffle_buffer_size=shuffle_buffer_size,
                local_shuffle_seed=shuffle_seed,
            )
        else:
            # https://docs.ray.io/en/latest/data/api/dataset.html#ray.data.Dataset.iter_batches
            return ray_data.iter_batches(
                batch_size=batch_size,
                prefetch_blocks=prefetch_blocks or self.DEFAULT_PREFETCH_BLOCKS,
                batch_format=self.stream_format,
                drop_last=drop_last,
                local_shuffle_buffer_size=shuffle_buffer_size,
                local_shuffle_seed=shuffle_seed,
            )

    def read_ray_dataset(self, columns: Optional[List[str]] = None):
        """Read parquet data as a ray dataset object"""
        # https://docs.ray.io/en/latest/data/api/input_output.html#parquet
        dataset = ray.data.read_parquet(
            self.fsspec_url, columns=columns, filesystem=self._folder.fsspec_fs
        )
        return dataset

    def write_ray_dataset(self, data_to_write: ray.data.Dataset):
        """Write a ray dataset to a fsspec filesystem"""
        if self.partition_cols:
            # TODO [JL]: https://arrow.apache.org/docs/python/generated/pyarrow.parquet.write_to_dataset.html
            logger.warning("Partitioning by column not currently supported.")
            pass

        # https://docs.ray.io/en/master/data/api/doc/ray.data.Dataset.write_parquet.html
        data_to_write.write_parquet(self.fsspec_url, filesystem=self._folder.fsspec_fs)

    @staticmethod
    def ray_dataset_from_arrow(data):
        """Convert an Arrow Table to a ray Dataset"""
        return ray.data.from_arrow(data)

    def delete_in_fs(self, recursive: bool = True):
        """Remove contents of all subdirectories (ex: partitioned data folders)"""
        # If file(s) are directories, recursively delete contents and then also remove the directory
        # https://filesystem-spec.readthedocs.io/en/latest/api.html#fsspec.spec.AbstractFileSystem.rm
        self._folder.fsspec_fs.rm(self.url, recursive=recursive)

    def exists_in_fs(self):
        return self._folder.exists_in_fs() and len(self._folder.ls(self.fsspec_url)) > 1

    def from_cluster(self, cluster):
        """Create a remote folder from a url on a cluster. This will create a virtual link into the
        cluster's filesystem. If you want to create a local copy or mount of the folder, use
        `Folder(url=<local_url>).sync_from_cluster(<cluster>, <url>)` or
        `Folder('url').from_cluster(<cluster>).mount(<local_url>)`."""
        if not cluster.address:
            raise ValueError("Cluster must be started before copying data from it.")
        new_table = copy.deepcopy(self)
        new_table._folder.fs = cluster
        return new_table


def _load_table_subclass(data, config: dict, dryrun: bool):
    """Load the relevant Table subclass based on the config or data type provided"""
    resource_subtype = config.get("resource_subtype", Table.__name__)

    try:
        import datasets

        if isinstance(data, datasets.Dataset) or resource_subtype == "HuggingFaceTable":
            from .huggingface_table import HuggingFaceTable

            return HuggingFaceTable.from_config(config)
    except ModuleNotFoundError:
        pass
    except Exception as e:
        raise e

    try:
        if isinstance(data, pd.DataFrame) or resource_subtype == "PandasTable":
            from .pandas_table import PandasTable

            return PandasTable.from_config(config)
    except ModuleNotFoundError:
        pass
    except Exception as e:
        raise e

    try:
        import dask.dataframe as dd

        if isinstance(data, dd.DataFrame) or resource_subtype == "DaskTable":
            from .dask_table import DaskTable

            return DaskTable.from_config(config)
    except ModuleNotFoundError:
        pass
    except Exception as e:
        raise e

    try:
        import ray

        if isinstance(data, ray.data.Dataset) or resource_subtype == "RayTable":
            from .ray_table import RayTable

            return RayTable.from_config(config)
    except ModuleNotFoundError:
        pass
    except Exception as e:
        raise e

    try:
        import cudf

        if isinstance(data, cudf.DataFrame) or resource_subtype == "CudfTable":
            # TODO [JL]
            # from .rapids_table import RapidsTable

            # return RapidsTable.from_config(config)
            raise NotImplementedError("Cudf not currently supported")
    except ModuleNotFoundError:
        pass
    except Exception as e:
        raise e

    if isinstance(data, pa.Table) or resource_subtype == "Table":
        new_table = Table.from_config(config, dryrun=dryrun)
        return new_table
    else:
        raise TypeError(
            f"Unsupported data type {type(data)} for Table construction. "
            f"For converting data to pyarrow see: "
            f"https://arrow.apache.org/docs/7.0/python/generated/pyarrow.Table.html"
        )


def table(
    data=None,
    name: Optional[str] = None,
    url: Optional[str] = None,
    fs: Optional[str] = None,
    data_config: Optional[dict] = None,
    partition_cols: Optional[list] = None,
    mkdir: bool = False,
    dryrun: bool = False,
    stream_format: Optional[str] = None,
    metadata: Optional[dict] = None,
):
    """Returns a Table object, which can be used to interact with the table at the given url.
    If the table does not exist, it will be saved if `dryrun` is False.
    """
    config = rns_client.load_config(name)

    config["fs"] = fs or config.get("fs") or rns_client.DEFAULT_FS
    if isinstance(config["fs"], str) and rns_client.exists(
        config["fs"], resource_type="cluster"
    ):
        config["fs"] = rns_client.load_config(config["fs"])

    name = name or config.get("rns_address") or config.get("name")
    name = name.lstrip("/") if name is not None else name

    data_url = url or config.get("url")
    file_name = None
    if data_url:
        # Extract the file name from the url if provided
        full_path = Path(data_url)
        file_suffix = full_path.suffix
        if file_suffix:
            data_url = str(full_path.parent)
            file_name = full_path.name

    if data_url is None:
        # TODO [JL] move some of the default params in this factory method to the defaults module for configurability
        if config["fs"] == rns_client.DEFAULT_FS:
            # create random url to store in .cache folder of local filesystem
            data_url = str(
                Path(
                    f"~/{Table.DEFAULT_CACHE_FOLDER}/{name or uuid.uuid4().hex}"
                ).expanduser()
            )
        else:
            # save to the default bucket
            data_url = f"{Table.DEFAULT_FOLDER_PATH}/{name}"

    config["name"] = name
    config["url"] = data_url
    config["file_name"] = file_name or config.get("file_name")
    config["data_config"] = data_config or config.get("data_config")
    config["partition_cols"] = partition_cols or config.get("partition_cols")
    config["stream_format"] = stream_format or config.get("stream_format")
    config["metadata"] = metadata or config.get("metadata")

    if mkdir:
        # create the remote folder for the table
        rh.folder(url=data_url, fs=fs, dryrun=True).mkdir()

    new_table = _load_table_subclass(data, config, dryrun)
    if data is not None:
        new_table.data = data

    return new_table
