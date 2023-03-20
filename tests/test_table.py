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
BUCKET_NAME = "runhouse-table"
NUM_PARTITIONS = 10


def setup():
    from runhouse.rns.api_utils.utils import create_s3_bucket

    create_s3_bucket(BUCKET_NAME)


def delete_local_folder(path):
    shutil.rmtree(path)


def tokenize_function(examples):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
    return tokenizer(examples["text"], padding="max_length", truncation=True)


def load_sample_data(data_type):
    if data_type == "huggingface":
        from datasets import load_dataset

        dataset = load_dataset("yelp_review_full", split="train[:1%]")
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
    local_path = Path.cwd() / "table_tests/local_test_table"
    local_path.mkdir(parents=True, exist_ok=True)

    Path(local_path).mkdir(parents=True, exist_ok=True)

    orig_data = pd.DataFrame({"my_col": list(range(50))})
    name = "~/my_local_test_table"

    my_table = (
        rh.table(
            data=orig_data,
            name=name,
            path=str(local_path),
            system="file",
        )
        .write()
        .save()
    )

    reloaded_table = rh.Table.from_name(name)
    reloaded_data: ray.data.Dataset = reloaded_table.data

    assert reloaded_data.to_pandas().equals(orig_data)

    batches = reloaded_table.stream(batch_size=10)
    for idx, batch in enumerate(batches):
        assert isinstance(batch, pd.DataFrame)
        assert batch["my_col"].tolist() == list(range(idx * 10, (idx + 1) * 10))

    del orig_data
    del my_table

    reloaded_table.delete_configs()

    reloaded_table.delete_in_system()
    assert not reloaded_table.exists_in_system()


def test_create_and_reload_pandas_locally():
    orig_data = load_sample_data("pandas")
    name = "~/my_test_local_pandas_table"

    my_table = (
        rh.table(
            data=orig_data,
            name=name,
            system="file",
            mkdir=True,
        )
        .write()
        .save()
    )

    reloaded_table = rh.Table.from_name(name)
    reloaded_data: ray.data.Dataset = reloaded_table.data

    assert orig_data.equals(reloaded_data.to_pandas())

    batches = reloaded_table.stream(batch_size=10)
    for idx, batch in enumerate(batches):
        assert isinstance(batch, pd.DataFrame)
        assert batch["id"].tolist() == list(range(1, 7))

    del orig_data
    del my_table

    reloaded_table.delete_configs()

    reloaded_table.delete_in_system()
    assert not reloaded_table.exists_in_system()


def test_create_and_reload_pyarrow_locally():
    orig_data = load_sample_data("pyarrow")
    name = "~/my_test_local_pyarrow_table"

    my_table = (
        rh.table(
            data=orig_data,
            name=name,
            system="file",
            mkdir=True,
        )
        .write()
        .save()
    )

    reloaded_table = rh.Table.from_name(name)
    reloaded_data: ray.data.Dataset = reloaded_table.data

    assert orig_data.to_pandas().equals(reloaded_data.to_pandas())

    batches = reloaded_table.stream(batch_size=10)
    for idx, batch in enumerate(batches):
        assert isinstance(batch, pa.Table)
        assert batch["int"].to_pylist() == list(range(1, 11))

    del orig_data
    del my_table

    reloaded_table.delete_configs()

    reloaded_table.delete_in_system()
    assert not reloaded_table.exists_in_system()


def test_create_and_reload_ray_locally():
    orig_data = load_sample_data("ray")
    name = "~/my_test_local_ray_table"

    my_table = (
        rh.table(
            data=orig_data,
            name=name,
            system="file",
            mkdir=True,
        )
        .write()
        .save()
    )

    reloaded_table = rh.Table.from_name(name)
    reloaded_data: ray.data.Dataset = reloaded_table.data

    assert orig_data.to_pandas().equals(reloaded_data.to_pandas())

    batches = reloaded_table.stream(batch_size=10)
    for idx, batch in enumerate(batches):
        assert isinstance(batch, pa.Table)
        assert batch["value"].to_pylist() == list(range(idx * 10, (idx + 1) * 10))

    del orig_data
    del my_table

    reloaded_table.delete_configs()

    reloaded_table.delete_in_system()
    assert not reloaded_table.exists_in_system()


def test_create_and_reload_huggingface_locally():
    orig_data = load_sample_data("huggingface")
    name = "~/my_test_local_huggingface_table"

    my_table = (
        rh.table(
            data=orig_data,
            name=name,
            system="file",
            mkdir=True,
        )
        .write()
        .save()
    )

    reloaded_table = rh.Table.from_name(name)
    reloaded_data: ray.data.Dataset = reloaded_table.data

    assert orig_data.to_pandas().equals(reloaded_data.to_pandas())

    batches = reloaded_table.stream(batch_size=10, as_dict=False)
    for idx, batch in enumerate(batches):
        assert batch.column_names == ["label", "text"]
        assert batch.shape == (10, 2)

    del orig_data
    del my_table

    reloaded_table.delete_configs()

    reloaded_table.delete_in_system()
    assert not reloaded_table.exists_in_system()


def test_create_and_reload_dask_locally():
    orig_data = load_sample_data("dask")
    name = "~/my_test_local_dask_table"

    my_table = (
        rh.table(
            data=orig_data,
            name=name,
            path="table_tests/dask_test_table",
            system="file",
            mkdir=True,
        )
        .write()
        .save()
    )

    reloaded_table = rh.Table.from_name(name)
    reloaded_data: "dask.dataframe.core.DataFrame" = reloaded_table.data.to_dask()
    assert reloaded_data.columns.to_list() == ["a", "b"]

    batches = reloaded_table.stream(batch_size=10)
    for idx, batch in enumerate(batches):
        assert batch.column_names == ["a", "b"]
        assert batch.shape == (10, 2)

    del orig_data
    del my_table

    reloaded_table.delete_configs()

    reloaded_table.delete_in_system()
    assert not reloaded_table.exists_in_system()


# --------------------------------------------
# ----------------- S3 tests -----------------
# --------------------------------------------
def test_create_and_reload_pyarrow_data_from_s3():
    orig_data = load_sample_data(data_type="pyarrow")
    name = "@/my_test_pyarrow_table"

    my_table = (
        rh.table(
            data=orig_data,
            name=name,
            path=f"/{BUCKET_NAME}/pyarrow_df",
            system="s3",
            mkdir=True,
        )
        .write()
        .save()
    )

    reloaded_table = rh.Table.from_name(name)
    reloaded_data: ray.data.Dataset = reloaded_table.data
    assert reloaded_data.to_pandas().equals(orig_data.to_pandas())

    batches = reloaded_table.stream(batch_size=10)
    for idx, batch in enumerate(batches):
        assert batch.column_names == ["int", "str"]

    del orig_data
    del my_table

    reloaded_table.delete_configs()

    reloaded_table.delete_in_system()
    assert not reloaded_table.exists_in_system()


def test_create_and_reload_pandas_data_from_s3():
    orig_data = load_sample_data(data_type="pandas")
    name = "@/my_test_pandas_table"

    my_table = (
        rh.table(
            data=orig_data,
            name=name,
            path=f"/{BUCKET_NAME}/pandas_df",
            system="s3",
            mkdir=True,
        )
        .write()
        .save()
    )

    reloaded_table = rh.Table.from_name(name)
    reloaded_data: ray.data.Dataset = reloaded_table.data
    assert orig_data.equals(reloaded_data.to_pandas())

    batches = reloaded_table.stream(batch_size=10)
    for idx, batch in enumerate(batches):
        assert isinstance(batch, pd.DataFrame)
        assert batch["id"].tolist() == list(range(1, 7))

    del orig_data
    del my_table

    reloaded_table.delete_configs()

    reloaded_table.delete_in_system()
    assert not reloaded_table.exists_in_system()


def test_create_and_reload_huggingface_data_from_s3():
    orig_data: datasets.Dataset = load_sample_data(data_type="huggingface")
    name = "@/my_test_hf_table"

    my_table = (
        rh.table(
            data=orig_data,
            name=name,
            path=f"/{BUCKET_NAME}/huggingface_data",
            system="s3",
            mkdir=True,
        )
        .write()
        .save()
    )

    reloaded_table = rh.Table.from_name(name)

    # Stream in as huggingface dataset
    batches = reloaded_table.stream(batch_size=10, as_dict=False)
    for idx, batch in enumerate(batches):
        assert batch.column_names == ["label", "text"]
        assert batch.shape == (10, 2)

    del orig_data
    del my_table

    reloaded_table.delete_configs()

    reloaded_table.delete_in_system()
    assert not reloaded_table.exists_in_system()


def test_create_and_reload_dask_data_from_s3():
    orig_data = load_sample_data(data_type="dask")
    name = "@/my_test_dask_table"

    my_table = (
        rh.table(
            data=orig_data,
            name=name,
            path=f"/{BUCKET_NAME}/dask",
            system="s3",
            mkdir=True,
        )
        .write()
        .save()
    )

    reloaded_table = rh.Table.from_name(name)
    reloaded_data: "dask.dataframe.core.DataFrame" = reloaded_table.data.to_dask()
    assert reloaded_data.columns.to_list() == ["a", "b"]

    batches = reloaded_table.stream(batch_size=10)
    for idx, batch in enumerate(batches):
        assert batch.column_names == ["a", "b"]
        assert batch.shape == (10, 2)

    del orig_data
    del my_table

    reloaded_table.delete_configs()

    reloaded_table.delete_in_system()
    assert not reloaded_table.exists_in_system()


def test_create_and_reload_ray_data_from_s3():
    orig_data = load_sample_data(data_type="ray")
    name = "@/my_test_ray_table"

    my_table = (
        rh.table(
            data=orig_data,
            name=name,
            path=f"/{BUCKET_NAME}/ray_data",
            system="s3",
            mkdir=True,
        )
        .write()
        .save()
    )

    reloaded_table = rh.Table.from_name(name)
    reloaded_data: ray.data.Dataset = reloaded_table.data
    assert reloaded_data.to_pandas().equals(orig_data.to_pandas())

    batches = reloaded_table.stream(batch_size=10)
    for idx, batch in enumerate(batches):
        assert isinstance(batch, pa.Table)
        assert batch["value"].to_pylist() == list(range(idx * 10, (idx + 1) * 10))

    del orig_data
    del my_table

    reloaded_table.delete_configs()

    reloaded_table.delete_in_system()
    assert not reloaded_table.exists_in_system()


# ----------------- Iter -----------------
def test_load_pandas_data_as_iter():
    orig_data = load_sample_data(data_type="pandas")
    name = "@/my_test_pandas_table"

    my_table = (
        rh.table(
            data=orig_data,
            name=name,
            path=f"/{BUCKET_NAME}/pandas",
            system="s3",
            mkdir=True,
        )
        .write()
        .save()
    )

    reloaded_table = rh.Table.from_name(name)
    reloaded_data: ray.data.Dataset = next(iter(reloaded_table))

    assert isinstance(reloaded_data, pd.Series)
    assert reloaded_data.to_dict() == {"id": 1, "grade": "a"}

    del orig_data
    del my_table

    reloaded_table.delete_configs()

    reloaded_table.delete_in_system()
    assert not reloaded_table.exists_in_system()


def test_load_pyarrow_data_as_iter():
    orig_data = load_sample_data(data_type="pyarrow")
    name = "@/my_test_pyarrow_table"

    my_table = (
        rh.table(
            data=orig_data,
            name=name,
            path=f"/{BUCKET_NAME}/pyarrow-data",
            system="s3",
            mkdir=True,
        )
        .write()
        .save()
    )

    reloaded_table = rh.Table.from_name(name)
    reloaded_data: pa.ChunkedArray = next(iter(reloaded_table))

    assert isinstance(reloaded_data, pa.ChunkedArray)
    assert reloaded_data.to_pylist() == list(range(1, 11))

    del orig_data
    del my_table

    reloaded_table.delete_configs()

    reloaded_table.delete_in_system()
    assert not reloaded_table.exists_in_system()


def test_load_huggingface_data_as_iter():
    orig_data = load_sample_data(data_type="huggingface")
    name = "@/my_test_huggingface_table"

    my_table = (
        rh.table(
            data=orig_data,
            name=name,
            path=f"/{BUCKET_NAME}/huggingface-dataset",
            system="s3",
            mkdir=True,
        )
        .write()
        .save()
    )

    reloaded_table = rh.Table.from_name(name)
    reloaded_data: pa.ChunkedArray = next(iter(reloaded_table))
    assert isinstance(reloaded_data, pa.ChunkedArray)

    del orig_data
    del my_table

    reloaded_table.delete_configs()

    reloaded_table.delete_in_system()
    assert not reloaded_table.exists_in_system()


# ----------------- Shuffling -----------------
def test_shuffling_pyarrow_data_from_s3():
    orig_data = load_sample_data(data_type="pyarrow")
    name = "@/my_test_shuffled_pyarrow_table"

    my_table = (
        rh.table(
            data=orig_data,
            name=name,
            path=f"/{BUCKET_NAME}/pyarrow",
            system="s3",
            mkdir=True,
        )
        .write()
        .save()
    )

    reloaded_table = rh.Table.from_name(name)
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

    reloaded_table.delete_in_system()
    assert not reloaded_table.exists_in_system()


# -------------------------------------------------
# ----------------- Cluster tests -----------------
# -------------------------------------------------
def test_create_and_reload_pandas_data_from_cluster():
    cluster = rh.cluster(name="^rh-cpu").up_if_not()

    # Make sure the destination folder for the data exists on the cluster
    data_path_on_cluster = f"{Folder.DEFAULT_CACHE_FOLDER}/pandas-data"
    cluster.run([f"mkdir -p {data_path_on_cluster}"])

    orig_data = load_sample_data(data_type="pandas")
    name = "@/my_test_pandas_table"
    my_table = (
        rh.table(
            data=orig_data,
            name=name,
            path=data_path_on_cluster,
            system=cluster,
        )
        .write()
        .save()
    )

    reloaded_table = rh.Table.from_name(name)

    reloaded_data: ray.data.Dataset = reloaded_table.data
    assert orig_data.equals(reloaded_data.to_pandas())

    batches = reloaded_table.stream(batch_size=10)
    for idx, batch in enumerate(batches):
        assert isinstance(batch, pd.DataFrame)
        assert batch["id"].tolist() == list(range(1, 7))

    del orig_data
    del my_table

    reloaded_table.delete_configs()

    reloaded_table.delete_in_system()
    assert not reloaded_table.exists_in_system()


def test_create_and_reload_ray_data_from_cluster():
    cluster = rh.cluster("^rh-cpu").up_if_not().save()

    data_path_on_cluster = f"{Folder.DEFAULT_CACHE_FOLDER}/ray-data"
    cluster.run([f"mkdir -p {data_path_on_cluster}"])

    orig_data = load_sample_data(data_type="ray")
    name = "@/my_test_ray_cluster_table"

    my_table = (
        rh.table(
            data=orig_data,
            name=name,
            path=data_path_on_cluster,
            system=cluster,
        )
        .write()
        .save()
    )

    reloaded_table = rh.Table.from_name(name)
    reloaded_data: ray.data.Dataset = reloaded_table.data
    assert orig_data.to_pandas().equals(reloaded_data.to_pandas())

    batches = reloaded_table.stream(batch_size=10)
    for idx, batch in enumerate(batches):
        assert isinstance(batch, pa.Table)
        assert batch["value"].to_pylist() == list(range(idx * 10, (idx + 1) * 10))

    del orig_data
    del my_table

    reloaded_table.delete_configs()

    reloaded_table.delete_in_system()
    assert not reloaded_table.exists_in_system()


def test_create_and_reload_pyarrow_data_from_cluster():
    cluster = rh.cluster("^rh-cpu").up_if_not().save()

    data_path_on_cluster = f"{Folder.DEFAULT_CACHE_FOLDER}/pyarrow-data"
    cluster.run([f"mkdir -p {data_path_on_cluster}"])

    orig_data = load_sample_data(data_type="pyarrow")
    name = "@/my_test_pyarrow_cluster_table"

    my_table = (
        rh.table(
            data=orig_data,
            name=name,
            path=data_path_on_cluster,
            system=cluster,
        )
        .write()
        .save()
    )

    reloaded_table = rh.Table.from_name(name)
    reloaded_data: ray.data.Dataset = reloaded_table.data
    assert orig_data.to_pandas().equals(reloaded_data.to_pandas())

    batches = reloaded_table.stream(batch_size=10)
    for idx, batch in enumerate(batches):
        assert batch.column_names == ["int", "str"]

    del orig_data
    del my_table

    reloaded_table.delete_configs()

    reloaded_table.delete_in_system()
    assert not reloaded_table.exists_in_system()


def test_create_and_reload_huggingface_data_from_cluster():
    cluster = rh.cluster("^rh-cpu").up_if_not().save()

    data_path_on_cluster = f"{Folder.DEFAULT_CACHE_FOLDER}/hf-data"
    cluster.run([f"mkdir -p {data_path_on_cluster}"])

    orig_data = load_sample_data(data_type="huggingface")
    name = "@/my_test_hf_cluster_table"

    my_table = (
        rh.table(
            data=orig_data,
            name=name,
            path=data_path_on_cluster,
            system=cluster,
        )
        .write()
        .save()
    )

    reloaded_table = rh.Table.from_name(name)
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

    reloaded_table.delete_in_system()
    assert not reloaded_table.exists_in_system()


def test_create_and_reload_dask_data_from_cluster():
    cluster = rh.cluster("^rh-cpu").up_if_not().save()

    data_path_on_cluster = f"{Folder.DEFAULT_CACHE_FOLDER}/dask-data"
    cluster.run([f"mkdir -p {data_path_on_cluster}"])

    orig_data = load_sample_data(data_type="dask")
    name = "@/my_test_dask_cluster_table"

    my_table = (
        rh.table(
            data=orig_data,
            name=name,
            path=data_path_on_cluster,
            system=cluster,
        )
        .write()
        .save()
    )

    reloaded_table = rh.Table.from_name(name)
    reloaded_data: "dask.dataframe.core.DataFrame" = reloaded_table.data.to_dask()
    assert reloaded_data.columns.to_list() == ["a", "b"]

    batches = reloaded_table.stream(batch_size=10)
    for idx, batch in enumerate(batches):
        assert batch.column_names == ["a", "b"]
        assert batch.shape == (10, 2)

    del orig_data
    del my_table

    reloaded_table.delete_configs()

    reloaded_table.delete_in_system()
    assert not reloaded_table.exists_in_system()


def test_to_cluster_attr():
    local_path = Path.cwd() / "table_tests/local_test_table"
    local_path.mkdir(parents=True, exist_ok=True)

    Path(local_path).mkdir(parents=True, exist_ok=True)

    orig_data = pd.DataFrame({"my_col": list(range(50))})
    name = "~/my_local_test_table"

    my_table = (
        rh.table(
            data=orig_data,
            name=name,
            path=str(local_path),
            system="file",
        )
        .write()
        .save()
    )

    cluster = rh.cluster("^rh-cpu").up_if_not().save()
    cluster_table = my_table.to(system=cluster)

    assert isinstance(cluster_table.system, rh.Cluster)
    assert cluster_table._folder._fs_str == "ssh"

    data = cluster_table.data
    assert data.to_pandas().equals(orig_data)

    del orig_data
    del my_table

    cluster_table.delete_configs()
    cluster_table.delete_in_system()
    assert not cluster_table.exists_in_system()


# -------------------------------------------------
# ----------------- Fetching tests -----------------
# -------------------------------------------------
def test_create_and_fetch_pyarrow_data_from_s3():
    orig_data = load_sample_data(data_type="pyarrow")
    name = "@/my_test_fetch_pyarrow_table"

    my_table = (
        rh.table(
            data=orig_data,
            name=name,
            path=f"/{BUCKET_NAME}/pyarrow",
            system="s3",
            mkdir=True,
        )
        .write()
        .save()
    )

    reloaded_table = rh.Table.from_name(name)
    reloaded_data: pa.Table = reloaded_table.fetch()
    assert orig_data == reloaded_data

    del orig_data
    del my_table

    reloaded_table.delete_configs()

    reloaded_table.delete_in_system()
    assert not reloaded_table.exists_in_system()


def test_create_and_fetch_pandas_data_from_s3():
    orig_data = load_sample_data(data_type="pandas")
    name = "@/my_test_fetch_pandas_table"

    my_table = (
        rh.table(
            data=orig_data,
            name=name,
            path=f"/{BUCKET_NAME}/pandas",
            system="s3",
            mkdir=True,
        )
        .write()
        .save()
    )

    reloaded_table = rh.Table.from_name(name)
    reloaded_data: pd.DataFrame = reloaded_table.fetch()
    assert orig_data.equals(reloaded_data)

    del orig_data
    del my_table

    reloaded_table.delete_configs()

    reloaded_table.delete_in_system()
    assert not reloaded_table.exists_in_system()


def test_create_and_fetch_huggingface_data_from_s3():
    orig_data = load_sample_data(data_type="huggingface")
    name = "@/my_test_fetch_huggingface_table"

    my_table = (
        rh.table(
            data=orig_data,
            name=name,
            path=f"/{BUCKET_NAME}/huggingface",
            system="s3",
            mkdir=True,
        )
        .write()
        .save()
    )

    reloaded_table = rh.Table.from_name(name)
    reloaded_data: datasets.Dataset = reloaded_table.fetch()
    assert orig_data.description == reloaded_data.description

    del orig_data
    del my_table

    reloaded_table.delete_configs()

    reloaded_table.delete_in_system()
    assert not reloaded_table.exists_in_system()


def test_create_and_fetch_ray_data_from_s3():
    orig_data = load_sample_data(data_type="ray")
    name = "@/my_test_fetch_ray_table"

    my_table = (
        rh.table(
            data=orig_data,
            name=name,
            path=f"/{BUCKET_NAME}/ray",
            system="s3",
            mkdir=True,
        )
        .write()
        .save()
    )

    reloaded_table = rh.Table.from_name(name)
    reloaded_data: ray.data.Dataset = reloaded_table.fetch()
    assert orig_data.to_pandas().equals(reloaded_data.to_pandas())

    del orig_data
    del my_table

    reloaded_table.delete_configs()

    reloaded_table.delete_in_system()
    assert not reloaded_table.exists_in_system()


def test_create_and_fetch_dask_data_from_s3():
    orig_data = load_sample_data(data_type="dask")
    name = "@/my_test_fetch_dask_table"
    my_table = (
        rh.table(
            data=orig_data,
            name=name,
            path=f"/{BUCKET_NAME}/dask",
            system="s3",
            mkdir=True,
        )
        .write()
        .save()
    )

    reloaded_table = rh.Table.from_name(name)
    reloaded_data: "dask.dataframe.core.DataFrame" = reloaded_table.fetch()
    assert orig_data.npartitions == reloaded_data.npartitions

    del orig_data
    del my_table

    reloaded_table.delete_configs()

    reloaded_table.delete_in_system()
    assert not reloaded_table.exists_in_system()


# -------------------------------------------------
# ----------------- Table Sharing tests -----------------
# -------------------------------------------------
def test_sharing_table():
    orig_data = load_sample_data(data_type="pandas")
    name = "shared_pandas_table"

    my_table = (
        rh.table(
            data=orig_data,
            name=name,
            system="s3",
            mkdir=True,
        )
        .write()
        .save()
    )

    my_table.share(
        users=["donny@run.house", "josh@run.house"],
        access_type="write",
        notify_users=False,
    )

    assert my_table.exists_in_system()


def test_read_shared_table():
    my_table = rh.Table.from_name(name="@/shared_pandas_table")
    df: pd.DataFrame = my_table.fetch()
    assert not df.empty


if __name__ == "__main__":
    setup()
    unittest.main()
