import pickle
from pathlib import Path

import pandas as pd

import pytest

import runhouse as rh

# https://docs.pytest.org/en/6.2.x/fixture.html#conftest-py-sharing-fixtures-across-multiple-files


@pytest.fixture
def blob_data():
    return pickle.dumps(list(range(50)))


@pytest.fixture
def local_folder():
    return rh.folder(path=Path.cwd() / "tests_tmp")


# ----------------- Tables -----------------
@pytest.fixture
def huggingface_table():
    from datasets import load_dataset

    dataset = load_dataset("yelp_review_full", split="train[:1%]")
    return dataset


@pytest.fixture
def arrow_table():
    import pyarrow as pa

    df = pd.DataFrame(
        {
            "int": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "str": ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"],
        }
    )
    arrow_table = pa.Table.from_pandas(df)
    return arrow_table


@pytest.fixture
def cudf_table():
    import cudf

    gdf = cudf.DataFrame(
        {"id": [1, 2, 3, 4, 5, 6], "grade": ["a", "b", "b", "a", "a", "e"]}
    )
    return gdf


@pytest.fixture
def pandas_table():
    df = pd.DataFrame(
        {"id": [1, 2, 3, 4, 5, 6], "grade": ["a", "b", "b", "a", "a", "e"]}
    )
    return df


@pytest.fixture
def dask_table():
    import dask.dataframe as dd

    index = pd.date_range("2021-09-01", periods=2400, freq="1H")
    df = pd.DataFrame({"a": range(2400), "b": list("abcaddbe" * 300)}, index=index)
    ddf = dd.from_pandas(df, npartitions=10)
    return ddf


@pytest.fixture
def ray_table():
    import ray

    ds = ray.data.range(10000)
    return ds


# ----------------- Clusters -----------------


@pytest.fixture
def cpu():
    return rh.cluster("^rh-cpu").up_if_not()


@pytest.fixture
def v100():
    return rh.cluster("^rh-v100").up_if_not()


@pytest.fixture
def k80():
    return rh.cluster("^rh-4-gpu").up_if_not()
