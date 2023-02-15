import logging
from typing import Optional

from .. import SkyCluster
from ..top_level_rns_fns import save
from .table import Table

logger = logging.getLogger(__name__)


class RapidsTable(Table):
    DEFAULT_FOLDER_PATH = "/runhouse/rapids-tables"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @staticmethod
    def from_config(config: dict, dryrun=True):
        """Load config values into the object."""
        if isinstance(config["fs"], dict):
            config["fs"] = SkyCluster.from_config(config["fs"], dryrun=dryrun)
        return RapidsTable(**config)

    def save(
        self,
        name: Optional[str] = None,
        snapshot: bool = False,
        overwrite: bool = True,
        **snapshot_kwargs,
    ):
        # https://docs.rapids.ai/api/cudf/nightly/api_docs/api/cudf.dataframe.to_parquet
        if self._cached_data is not None:
            self.data.to_parquet(self.fsspec_url)
            logger.info(f"Saved {str(self)} to: {self.fsspec_url}")

        save(self, name=name, snapshot=snapshot, overwrite=overwrite, **snapshot_kwargs)

        return self

    def fetch(self, **kwargs):
        import cudf

        # https://docs.rapids.ai/api/cudf/nightly/api_docs/api/cudf.read_parquet.html
        self._cached_data = cudf.read_parquet(self.fsspec_url)
        return self._cached_data
