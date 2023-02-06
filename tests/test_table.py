import shutil
import unittest
from pathlib import Path

import datasets
import pandas as pd
import pyarrow as pa
import ray.data

import runhouse as rh
from runhouse import Folder

TEMP_LOCAL_FOLDER = Path("~/.rh/temp").expanduser()
BUCKET_NAME = "runhouse-tests"
NUM_PARTITIONS = 10


def setup():
    # Create bucket in S3
    from sky.data.storage import S3Store

    S3Store(name=BUCKET_NAME, source="")


def delete_local_folder(path):
    shutil.rmtree(path)


def tokenize_function(examples):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
    return tokenizer(examples["text"], padding="max_length", truncation=True)


def load_sample_data(data_type):
    if data_type == "huggingface":
        from datasets import load_dataset

        dataset = load_dataset("yelp_review_full", split="train[:10%]")
        return dataset

    elif data_type == "pyarrow":
        df = pd.DataFrame(
            {
                "int": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                "str": ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"],
            }
        )
        arrow_table = pa.Table.from_pandas(df)
        return arrow_table

    elif data_type == "cudf":
        import cudf

        gdf = cudf.DataFrame(
            {"id": [1, 2, 3, 4, 5, 6], "grade": ["a", "b", "b", "a", "a", "e"]}
        )
        return gdf

    elif data_type == "pandas":
        df = pd.DataFrame(
            {"id": [1, 2, 3, 4, 5, 6], "grade": ["a", "b", "b", "a", "a", "e"]}
        )
        return df

    elif data_type == "dask":
        import dask.dataframe as dd

        index = pd.date_range("2021-09-01", periods=2400, freq="1H")
        df = pd.DataFrame({"a": range(2400), "b": list("abcaddbe" * 300)}, index=index)
        ddf = dd.from_pandas(df, npartitions=NUM_PARTITIONS)
        return ddf

    elif data_type == "ray":
        import ray

        ds = ray.data.range(10000)
        return ds

    else:
        raise Exception(f"Unsupported data type {data_type}")


# -----------------------------------------------
# ----------------- Local tests -----------------
# -----------------------------------------------


def test_create_and_reload_file_locally():
    local_url = Path.cwd() / "table_tests/local_test_table"
    local_url.mkdir(parents=True, exist_ok=True)

    Path(local_url).mkdir(parents=True, exist_ok=True)

    orig_data = pd.DataFrame({"my_col": list(range(50))})
    my_table = rh.table(
        data=orig_data, name="~/my_local_test_table", url=str(local_url), fs="file"
    ).save()

    reloaded_table = rh.table(name="~/my_local_test_table", dryrun=True)
    reloaded_data: ray.data.Dataset = reloaded_table.data

    assert reloaded_data.to_pandas().equals(orig_data)

    batches = reloaded_table.stream(batch_size=10)
    for idx, batch in enumerate(batches):
        assert isinstance(batch, pd.DataFrame)
        assert batch["my_col"].tolist() == list(range(idx * 10, (idx + 1) * 10))

    del orig_data
    del my_table

    reloaded_table.delete_configs()

    reloaded_table.delete_in_fs()
    assert not reloaded_table.exists_in_fs()


def test_create_and_reload_pandas_locally():
    orig_data = load_sample_data("pandas")

    my_table = rh.table(
        data=orig_data,
        name="~/my_test_local_pandas_table",
        url="table_tests/pandas_test_table",
        fs="file",
        mkdir=True,
    ).save()

    reloaded_table = rh.table(name="~/my_test_local_pandas_table", dryrun=True)
    reloaded_data: ray.data.Dataset = reloaded_table.data

    assert orig_data.equals(reloaded_data.to_pandas())

    batches = reloaded_table.stream(batch_size=10)
    for idx, batch in enumerate(batches):
        assert isinstance(batch, pd.DataFrame)
        assert batch["id"].tolist() == list(range(1, 7))

    del orig_data
    del my_table

    reloaded_table.delete_configs()

    reloaded_table.delete_in_fs()
    assert not reloaded_table.exists_in_fs()


def test_create_and_reload_pyarrow_locally():
    orig_data = load_sample_data("pyarrow")
    my_table = rh.table(
        data=orig_data,
        name="~/my_test_local_pyarrow_table",
        url="table_tests/pyarrow_test_table",
        fs="file",
        mkdir=True,
    ).save()

    reloaded_table = rh.table(name="~/my_test_local_pyarrow_table", dryrun=True)
    reloaded_data: ray.data.Dataset = reloaded_table.data

    assert orig_data.to_pandas().equals(reloaded_data.to_pandas())

    batches = reloaded_table.stream(batch_size=10)
    for idx, batch in enumerate(batches):
        assert isinstance(batch, pa.Table)
        assert batch["int"].to_pylist() == list(range(1, 11))

    del orig_data
    del my_table

    reloaded_table.delete_configs()

    reloaded_table.delete_in_fs()
    assert not reloaded_table.exists_in_fs()


def test_create_and_reload_ray_locally():
    orig_data = load_sample_data("ray")
    my_table = rh.table(
        data=orig_data,
        name="~/my_test_local_ray_table",
        url="table_tests/ray_test_table",
        fs="file",
        mkdir=True,
    ).save()

    reloaded_table = rh.table(name="~/my_test_local_ray_table", dryrun=True)
    reloaded_data: ray.data.Dataset = reloaded_table.data

    assert orig_data.to_pandas().equals(reloaded_data.to_pandas())

    batches = reloaded_table.stream(batch_size=10)
    for idx, batch in enumerate(batches):
        assert isinstance(batch, pa.Table)
        assert batch["value"].to_pylist() == list(range(idx * 10, (idx + 1) * 10))

    del orig_data
    del my_table

    reloaded_table.delete_configs()

    reloaded_table.delete_in_fs()
    assert not reloaded_table.exists_in_fs()


def test_create_and_reload_huggingface_locally():
    orig_data = load_sample_data("huggingface")
    my_table = rh.table(
        data=orig_data,
        name="~/my_test_local_huggingface_table",
        url="table_tests/huggingface_test_table",
        fs="file",
        mkdir=True,
    ).save()

    reloaded_table = rh.table(name="~/my_test_local_huggingface_table", dryrun=True)
    reloaded_data: ray.data.Dataset = reloaded_table.data

    assert orig_data.to_pandas().equals(reloaded_data.to_pandas())

    batches = reloaded_table.stream(batch_size=10, as_dict=False)
    for idx, batch in enumerate(batches):
        assert batch.column_names == ["label", "text"]
        assert batch.shape == (10, 2)

    del orig_data
    del my_table

    reloaded_table.delete_configs()

    reloaded_table.delete_in_fs()
    assert not reloaded_table.exists_in_fs()


def test_create_and_reload_dask_locally():
    orig_data = load_sample_data("dask")
    my_table = rh.table(
        data=orig_data,
        name="~/my_test_local_dask_table",
        url="table_tests/dask_test_table",
        fs="file",
        mkdir=True,
    ).save()

    reloaded_table = rh.table(name="~/my_test_local_dask_table", dryrun=True)
    reloaded_data: "dask.dataframe.core.DataFrame" = reloaded_table.data.to_dask()
    assert reloaded_data.columns.to_list() == ["a", "b"]

    batches = reloaded_table.stream(batch_size=10)
    for idx, batch in enumerate(batches):
        assert batch.column_names == ["a", "b"]
        assert batch.shape == (10, 2)

    del orig_data
    del my_table

    reloaded_table.delete_configs()

    reloaded_table.delete_in_fs()
    assert not reloaded_table.exists_in_fs()


# --------------------------------------------
# ----------------- S3 tests -----------------
# --------------------------------------------
def test_create_and_reload_pyarrow_data_from_s3():
    orig_data = load_sample_data(data_type="pyarrow")

    my_table = rh.table(
        data=orig_data,
        name="@/my_test_pyarrow_table",
        url=f"/{BUCKET_NAME}/pyarrow_df",
        fs="s3",
        mkdir=True,
    ).save()

    reloaded_table = rh.table(name="@/my_test_pyarrow_table", dryrun=True)
    reloaded_data: ray.data.Dataset = reloaded_table.data
    assert reloaded_data.to_pandas().equals(orig_data.to_pandas())

    batches = reloaded_table.stream(batch_size=10)
    for idx, batch in enumerate(batches):
        assert batch.column_names == ["int", "str"]

    del orig_data
    del my_table

    reloaded_table.delete_configs()

    reloaded_table.delete_in_fs()
    assert not reloaded_table.exists_in_fs()


def test_create_and_reload_pandas_data_from_s3():
    orig_data = load_sample_data(data_type="pandas")

    my_table = rh.table(
        data=orig_data,
        name="@/my_test_pandas_table",
        url=f"/{BUCKET_NAME}/pandas_df",
        fs="s3",
        mkdir=True,
    ).save()

    reloaded_table = rh.table(name="@/my_test_pandas_table", dryrun=True)
    reloaded_data: ray.data.Dataset = reloaded_table.data
    assert orig_data.equals(reloaded_data.to_pandas())

    batches = reloaded_table.stream(batch_size=10)
    for idx, batch in enumerate(batches):
        assert isinstance(batch, pd.DataFrame)
        assert batch["id"].tolist() == list(range(1, 7))

    del orig_data
    del my_table

    reloaded_table.delete_configs()

    reloaded_table.delete_in_fs()
    assert not reloaded_table.exists_in_fs()


def test_create_and_reload_huggingface_data_from_s3():
    orig_data: datasets.Dataset = load_sample_data(data_type="huggingface")

    my_table = rh.table(
        data=orig_data,
        name="@/my_test_hf_table",
        url=f"/{BUCKET_NAME}/huggingface_data",
        fs="s3",
        mkdir=True,
    ).save()

    reloaded_table = rh.table(name="@/my_test_hf_table", dryrun=True)

    # Stream in as huggingface dataset
    batches = reloaded_table.stream(batch_size=10, as_dict=False)
    for idx, batch in enumerate(batches):
        assert batch.column_names == ["label", "text"]
        assert batch.shape == (10, 2)

    del orig_data
    del my_table

    reloaded_table.delete_configs()

    reloaded_table.delete_in_fs()
    assert not reloaded_table.exists_in_fs()


def test_create_and_reload_dask_data_from_s3():
    orig_data = load_sample_data(data_type="dask")
    my_table = rh.table(
        data=orig_data,
        name="@/my_test_dask_table",
        url=f"/{BUCKET_NAME}/dask",
        fs="s3",
        mkdir=True,
    ).save()

    reloaded_table = rh.table(name="@/my_test_dask_table", dryrun=True)
    reloaded_data: "dask.dataframe.core.DataFrame" = reloaded_table.data.to_dask()
    assert reloaded_data.columns.to_list() == ["a", "b"]

    batches = reloaded_table.stream(batch_size=10)
    for idx, batch in enumerate(batches):
        assert batch.column_names == ["a", "b"]
        assert batch.shape == (10, 2)

    del orig_data
    del my_table

    reloaded_table.delete_configs()

    reloaded_table.delete_in_fs()
    assert not reloaded_table.exists_in_fs()


def test_create_and_reload_ray_data_from_s3():
    orig_data = load_sample_data(data_type="ray")

    my_table = rh.table(
        data=orig_data,
        name="@/my_test_ray_table",
        url=f"/{BUCKET_NAME}/ray_data",
        fs="s3",
        mkdir=True,
    ).save()

    reloaded_table = rh.table(name="@/my_test_ray_table", dryrun=True)
    reloaded_data: ray.data.Dataset = reloaded_table.data
    assert reloaded_data.to_pandas().equals(orig_data.to_pandas())

    batches = reloaded_table.stream(batch_size=10)
    for idx, batch in enumerate(batches):
        assert isinstance(batch, pa.Table)
        assert batch["value"].to_pylist() == list(range(idx * 10, (idx + 1) * 10))

    del orig_data
    del my_table

    reloaded_table.delete_configs()

    reloaded_table.delete_in_fs()
    assert not reloaded_table.exists_in_fs()


# ----------------- Iter -----------------
def test_load_pandas_data_as_iter():
    orig_data = load_sample_data(data_type="pandas")

    my_table = rh.table(
        data=orig_data,
        name="@/my_test_pandas_table",
        url=f"/{BUCKET_NAME}/pandas",
        fs="s3",
        mkdir=True,
    ).save()

    reloaded_table = rh.table(name="@/my_test_pandas_table", dryrun=True)
    reloaded_data: ray.data.Dataset = next(iter(reloaded_table))

    assert isinstance(reloaded_data, pd.Series)
    assert reloaded_data.to_dict() == {"id": 1, "grade": "a"}

    del orig_data
    del my_table

    reloaded_table.delete_configs()

    reloaded_table.delete_in_fs()
    assert not reloaded_table.exists_in_fs()


def test_load_pyarrow_data_as_iter():
    orig_data = load_sample_data(data_type="pyarrow")

    my_table = rh.table(
        data=orig_data,
        name="@/my_test_pyarrow_table",
        url=f"/{BUCKET_NAME}/pyarrow-data",
        fs="s3",
        mkdir=True,
    ).save()

    reloaded_table = rh.table(name="@/my_test_pyarrow_table", dryrun=True)
    reloaded_data: pa.ChunkedArray = next(iter(reloaded_table))

    assert isinstance(reloaded_data, pa.ChunkedArray)
    assert reloaded_data.to_pylist() == list(range(1, 11))

    del orig_data
    del my_table

    reloaded_table.delete_configs()

    reloaded_table.delete_in_fs()
    assert not reloaded_table.exists_in_fs()


def test_load_huggingface_data_as_iter():
    orig_data = load_sample_data(data_type="huggingface")

    my_table = rh.table(
        data=orig_data,
        name="@/my_test_huggingface_table",
        url=f"/{BUCKET_NAME}/huggingface-dataset",
        fs="s3",
        mkdir=True,
    ).save()

    reloaded_table = rh.table(name="@/my_test_huggingface_table", dryrun=True)
    reloaded_data: pa.ChunkedArray = next(iter(reloaded_table))
    assert isinstance(reloaded_data, pa.ChunkedArray)

    del orig_data
    del my_table

    reloaded_table.delete_configs()

    reloaded_table.delete_in_fs()
    assert not reloaded_table.exists_in_fs()


# ----------------- Shuffling -----------------
def test_shuffling_pyarrow_data_from_s3():
    orig_data = load_sample_data(data_type="pyarrow")

    my_table = rh.table(
        data=orig_data,
        name="@/my_test_shuffled_pyarrow_table",
        url=f"/{BUCKET_NAME}/pyarrow",
        fs="s3",
        mkdir=True,
    ).save()

    reloaded_table = rh.table(name="@/my_test_shuffled_pyarrow_table", dryrun=True)
    batches = reloaded_table.stream(
        batch_size=10, shuffle_seed=42, shuffle_buffer_size=10
    )
    for idx, batch in enumerate(batches):
        assert isinstance(batch, pa.Table)
        assert orig_data.columns[0].to_pylist() != batch.columns[0].to_pylist()
        assert orig_data.columns[1].to_pylist() != batch.columns[1].to_pylist()

    del orig_data
    del my_table

    reloaded_table.delete_configs()

    reloaded_table.delete_in_fs()
    assert not reloaded_table.exists_in_fs()


# -------------------------------------------------
# ----------------- Cluster tests -----------------
# -------------------------------------------------
def test_create_and_reload_pandas_data_from_cluster():
    cluster = rh.cluster(name="^rh-cpu").up_if_not().save()

    # Make sure the destination folder for the data exists on the cluster
    data_url_on_cluster = f"{Folder.DEFAULT_CACHE_FOLDER}/pandas-data"
    cluster.run([f"mkdir -p {data_url_on_cluster}"])

    orig_data = load_sample_data(data_type="pandas")

    my_table = rh.table(
        data=orig_data,
        name="@/my_test_pandas_table",
        url=data_url_on_cluster,
        fs=cluster,
    ).save()

    reloaded_table = rh.table(name="@/my_test_pandas_table", dryrun=True)

    reloaded_data: ray.data.Dataset = reloaded_table.data
    assert orig_data.equals(reloaded_data.to_pandas())

    batches = reloaded_table.stream(batch_size=10)
    for idx, batch in enumerate(batches):
        assert isinstance(batch, pd.DataFrame)
        assert batch["id"].tolist() == list(range(1, 7))

    del orig_data
    del my_table

    reloaded_table.delete_configs()

    reloaded_table.delete_in_fs()
    assert not reloaded_table.exists_in_fs()


def test_create_and_reload_ray_data_from_cluster():
    cluster = rh.cluster("^rh-cpu").up_if_not()

    data_url_on_cluster = f"{Folder.DEFAULT_CACHE_FOLDER}/ray-data"
    cluster.run([f"mkdir -p {data_url_on_cluster}"])

    orig_data = load_sample_data(data_type="ray")

    my_table = rh.table(
        data=orig_data,
        name="@/my_test_ray_cluster_table",
        url=data_url_on_cluster,
        fs=cluster,
    ).save()

    reloaded_table = rh.table(name="@/my_test_ray_cluster_table", dryrun=True)
    reloaded_data: ray.data.Dataset = reloaded_table.data
    assert orig_data.to_pandas().equals(reloaded_data.to_pandas())

    batches = reloaded_table.stream(batch_size=10)
    for idx, batch in enumerate(batches):
        assert isinstance(batch, pa.Table)
        assert batch["value"].to_pylist() == list(range(idx * 10, (idx + 1) * 10))

    del orig_data
    del my_table

    reloaded_table.delete_configs()

    reloaded_table.delete_in_fs()
    assert not reloaded_table.exists_in_fs()


def test_create_and_reload_pyarrow_data_from_cluster():
    cluster = rh.cluster("^rh-cpu").up_if_not()

    data_url_on_cluster = f"{Folder.DEFAULT_CACHE_FOLDER}/pyarrow-data"
    cluster.run([f"mkdir -p {data_url_on_cluster}"])

    orig_data = load_sample_data(data_type="pyarrow")

    my_table = rh.table(
        data=orig_data,
        name="@/my_test_pyarrow_cluster_table",
        url=data_url_on_cluster,
        fs=cluster,
    ).save()

    reloaded_table = rh.table(name="@/my_test_pyarrow_cluster_table", dryrun=True)
    reloaded_data: ray.data.Dataset = reloaded_table.data
    assert orig_data.to_pandas().equals(reloaded_data.to_pandas())

    batches = reloaded_table.stream(batch_size=10)
    for idx, batch in enumerate(batches):
        assert batch.column_names == ["int", "str"]

    del orig_data
    del my_table

    reloaded_table.delete_configs()

    reloaded_table.delete_in_fs()
    assert not reloaded_table.exists_in_fs()


def test_create_and_reload_huggingface_data_from_cluster():
    cluster = rh.cluster("^rh-cpu").up_if_not()

    data_url_on_cluster = f"{Folder.DEFAULT_CACHE_FOLDER}/hf-data"
    cluster.run([f"mkdir -p {data_url_on_cluster}"])

    orig_data = load_sample_data(data_type="huggingface")

    my_table = rh.table(
        data=orig_data,
        name="@/my_test_hf_cluster_table",
        url=data_url_on_cluster,
        fs=cluster,
    ).save()

    reloaded_table = rh.table(name="@/my_test_hf_cluster_table", dryrun=True)
    reloaded_data: ray.data.Dataset = reloaded_table.data
    assert orig_data.to_pandas().equals(reloaded_data.to_pandas())

    # Stream in as huggingface dataset
    batches = reloaded_table.stream(batch_size=10, as_dict=False)
    for idx, batch in enumerate(batches):
        assert batch.column_names == ["label", "text"]
        assert batch.shape == (10, 2)

    del orig_data
    del my_table

    reloaded_table.delete_configs()

    reloaded_table.delete_in_fs()
    assert not reloaded_table.exists_in_fs()


def test_create_and_reload_dask_data_from_cluster():
    cluster = rh.cluster("^rh-cpu").up_if_not()

    data_url_on_cluster = f"{Folder.DEFAULT_CACHE_FOLDER}/dask-data"
    cluster.run([f"mkdir -p {data_url_on_cluster}"])

    orig_data = load_sample_data(data_type="dask")

    my_table = rh.table(
        data=orig_data,
        name="@/my_test_dask_cluster_table",
        url=data_url_on_cluster,
        fs=cluster,
    ).save()

    reloaded_table = rh.table(name="@/my_test_dask_cluster_table", dryrun=True)
    reloaded_data: "dask.dataframe.core.DataFrame" = reloaded_table.data.to_dask()
    assert reloaded_data.columns.to_list() == ["a", "b"]

    batches = reloaded_table.stream(batch_size=10)
    for idx, batch in enumerate(batches):
        assert batch.column_names == ["a", "b"]
        assert batch.shape == (10, 2)

    del orig_data
    del my_table

    reloaded_table.delete_configs()

    reloaded_table.delete_in_fs()
    assert not reloaded_table.exists_in_fs()


if __name__ == "__main__":
    setup()
    unittest.main()
