import logging
import uuid
from typing import Optional

from .. import OnDemandCluster
from ..top_level_rns_fns import save

from .table import Table

logger = logging.getLogger(__name__)


class PandasTable(Table):
    DEFAULT_FOLDER_PATH = "/runhouse/pandas-tables"
    DEFAULT_STREAM_FORMAT = "pandas"

    def __init__(self, **kwargs):
        if not kwargs.get("file_name"):
            kwargs["file_name"] = f"{uuid.uuid4().hex}.parquet"
        super().__init__(**kwargs)

    def __iter__(self):
        for block in self.stream(batch_size=self.DEFAULT_BATCH_SIZE):
            for idx, row in block.iterrows():
                yield row

    @staticmethod
    def from_config(config: dict, dryrun=True):
        """Load config values into the object."""
        if isinstance(config["system"], dict):
            config["system"] = OnDemandCluster.from_config(
                config["system"], dryrun=dryrun
            )
        return PandasTable(**config, dryrun=dryrun)

    def save(
        self,
        name: Optional[str] = None,
        snapshot: bool = False,
        overwrite: bool = True,
        **snapshot_kwargs,
    ):
        if self._cached_data is not None:
            # https://pandas.pydata.org/pandas-docs/version/1.1/reference/api/pandas.DataFrame.to_parquet.html
            self.data.to_parquet(
                self.fsspec_url,
                storage_options=self.data_config,
                partition_cols=self.partition_cols,
            )

        save(self, name=name, snapshot=snapshot, overwrite=overwrite, **snapshot_kwargs)

        return self

    def fetch(self, **kwargs):
        import pandas as pd

        # https://pandas.pydata.org/docs/reference/api/pandas.read_parquet.html
        self._cached_data = pd.read_parquet(
            self.fsspec_url, storage_options=self.data_config
        )
        return self._cached_data
