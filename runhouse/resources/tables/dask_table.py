import logging

from .table import Table

logger = logging.getLogger(__name__)


class DaskTable(Table):
    DEFAULT_FOLDER_PATH = "/runhouse/dask-tables"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def write(self, write_index: bool = False):
        # https://docs.dask.org/en/stable/how-to/connect-to-remote-data.html
        if self._cached_data is not None:
            # https://stackoverflow.com/questions/72891631/how-to-remove-null-dask-index-from-parquet-file
            self.data.to_parquet(
                self.fsspec_url,
                write_index=write_index,
                storage_options=self.data_config,
            )
            logger.info(f"Saved {str(self)} to: {self.fsspec_url}")

        return self

    def fetch(self, **kwargs):
        import dask.dataframe as dd

        # https://docs.dask.org/en/stable/generated/dask.dataframe.read_parquet.html
        self._cached_data = dd.read_parquet(
            self.fsspec_url, storage_options=self.data_config
        )

        return self._cached_data
