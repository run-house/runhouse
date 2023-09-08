import logging

from .table import Table

logger = logging.getLogger(__name__)


class RapidsTable(Table):
    DEFAULT_FOLDER_PATH = "/runhouse/rapids-tables"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def write(self):
        # https://docs.rapids.ai/api/cudf/nightly/api_docs/api/cudf.dataframe.to_parquet
        if self._cached_data is not None:
            self.data.to_parquet(self.fsspec_url)
            logger.info(f"Saved {str(self)} to: {self.fsspec_url}")

        return self

    def fetch(self, **kwargs):
        import cudf

        # https://docs.rapids.ai/api/cudf/nightly/api_docs/api/cudf.read_parquet.html
        self._cached_data = cudf.read_parquet(self.fsspec_url)
        return self._cached_data
