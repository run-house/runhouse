import shutil

import pandas as pd
import pyarrow as pa
import ray.data
import runhouse as rh

from runhouse import Folder

NUM_PARTITIONS = 10


# TODO top to bottom update. Named "table_tests" so it's skipped until we add a proper test class and suite


def delete_local_folder(path):
    shutil.rmtree(path)


def tokenize_function(examples):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
    return tokenizer(examples["text"], padding="max_length", truncation=True)


# -----------------------------------------------
# ----------------- Local tests -----------------
# -----------------------------------------------
def test_create_and_reload_file_locally(tmp_path):
    orig_data = pd.DataFrame({"my_col": list(range(50))})
    name = "~/my_local_test_table"

    my_table = (
        rh.table(
            data=orig_data,
            name=name,
            path=str(tmp_path),
            system="file",
        )
        .write()
        .save()
    )

    reloaded_table = rh.table(name=name)
    reloaded_data: ray.data.Dataset = reloaded_table.data

    assert reloaded_data.to_pandas().equals(orig_data)

    batches = reloaded_table.stream(batch_size=10)
    for idx, batch in enumerate(batches):
        assert isinstance(batch, pd.DataFrame)
        assert batch["my_col"].tolist() == list(range(idx * 10, (idx + 1) * 10))

    del orig_data
    del my_table

    reloaded_table.delete_configs()

    reloaded_table.rm()
    assert not reloaded_table.exists_in_system()


def test_create_and_reload_pandas_locally(pandas_table, tmp_path):
    name = "~/my_test_local_pandas_table"

    my_table = (
        rh.table(
            data=pandas_table,
            path=str(tmp_path),
            name=name,
            system="file",
            mkdir=True,
        )
        .write()
        .save()
    )

    reloaded_table = rh.table(name=name)
    reloaded_data: ray.data.Dataset = reloaded_table.data

    assert pandas_table.equals(reloaded_data.to_pandas())

    batches = reloaded_table.stream(batch_size=10)
    for idx, batch in enumerate(batches):
        assert isinstance(batch, pd.DataFrame)
        assert batch["id"].tolist() == list(range(1, 7))

    del pandas_table
    del my_table

    reloaded_table.delete_configs()

    reloaded_table.rm()
    assert not reloaded_table.exists_in_system()


def test_create_and_reload_pyarrow_locally(arrow_table, tmp_path):
    name = "~/my_test_local_pyarrow_table"

    my_table = (
        rh.table(
            data=arrow_table,
            name=name,
            path=str(tmp_path),
            system="file",
            mkdir=True,
        )
        .write()
        .save()
    )

    reloaded_table = rh.table(name=name)
    reloaded_data: ray.data.Dataset = reloaded_table.data

    assert arrow_table.to_pandas().equals(reloaded_data.to_pandas())

    batches = reloaded_table.stream(batch_size=10)
    for idx, batch in enumerate(batches):
        assert isinstance(batch, pa.Table)
        assert batch["int"].to_pylist() == list(range(1, 11))

    del arrow_table
    del my_table

    reloaded_table.delete_configs()

    reloaded_table.rm()
    assert not reloaded_table.exists_in_system()


def test_create_and_reload_ray_locally(ray_table, tmp_path):
    name = "~/my_test_local_ray_table"

    my_table = (
        rh.table(
            data=ray_table,
            path=str(tmp_path),
            name=name,
            system="file",
            mkdir=True,
        )
        .write()
        .save()
    )

    reloaded_table = rh.table(name=name)
    reloaded_data: ray.data.Dataset = reloaded_table.data

    assert ray_table.to_pandas().equals(reloaded_data.to_pandas())

    batches = reloaded_table.stream(batch_size=10)
    for idx, batch in enumerate(batches):
        assert isinstance(batch, pa.Table)
        # NOTE [DG] 2021-08-10: This will generally fail because ray automatically partitions the data into
        # blocks, and order is not necessarily preserved when reading the data back in. Ideally we fix this
        # when we switch to in-memory tables.
        # assert batch["value"].to_pylist() == list(range(idx * 10, (idx + 1) * 10))

        if idx in [0, 10, 33]:
            # Some random batches to check
            assert [isinstance(val, int) for val in batch["value"].to_pylist()]

    del ray_table
    del my_table

    reloaded_table.delete_configs()

    reloaded_table.rm()
    assert not reloaded_table.exists_in_system()


def test_create_and_reload_huggingface_locally(huggingface_table, tmp_path):
    name = "~/my_test_local_huggingface_table"

    my_table = (
        rh.table(
            data=huggingface_table,
            name=name,
            path=str(tmp_path),
            system="file",
            mkdir=True,
        )
        .write()
        .save()
    )

    reloaded_table = rh.table(name=name)
    reloaded_data: ray.data.Dataset = reloaded_table.data

    assert huggingface_table.to_pandas().equals(reloaded_data.to_pandas())

    batches = reloaded_table.stream(batch_size=10, as_dict=False)
    for idx, batch in enumerate(batches):
        assert batch.column_names == ["label", "text"]
        assert batch.shape == (10, 2)

    del huggingface_table
    del my_table

    reloaded_table.delete_configs()

    reloaded_table.rm()
    assert not reloaded_table.exists_in_system()


def test_create_and_reload_dask_locally(dask_table, tmp_path):
    name = "~/my_test_local_dask_table"

    my_table = (
        rh.table(
            data=dask_table,
            name=name,
            path=str(tmp_path),
            system="file",
            mkdir=True,
        )
        .write()
        .save()
    )

    reloaded_table = rh.table(name=name)
    reloaded_data: "dask.dataframe.core.DataFrame" = reloaded_table.data.to_dask()
    assert reloaded_data.columns.to_list() == ["a", "b"]

    batches = reloaded_table.stream(batch_size=10)
    for idx, batch in enumerate(batches):
        assert batch.column_names == ["a", "b"]
        assert batch.shape == (10, 2)

    del dask_table
    del my_table

    reloaded_table.delete_configs()

    reloaded_table.rm()
    assert not reloaded_table.exists_in_system()


# --------------------------------------------
# ----------------- S3 tests -----------------
# --------------------------------------------


def test_create_and_reload_pyarrow_data_from_s3(arrow_table, table_s3_bucket):
    name = "@/my_test_pyarrow_table"

    my_table = (
        rh.table(
            data=arrow_table,
            name=name,
            path=f"/{table_s3_bucket}/pyarrow_df",
            system="s3",
            mkdir=True,
        )
        .write()
        .save()
    )

    reloaded_table = rh.table(name=name)
    reloaded_data: ray.data.Dataset = reloaded_table.data
    assert reloaded_data.to_pandas().equals(arrow_table.to_pandas())

    batches = reloaded_table.stream(batch_size=10)
    for idx, batch in enumerate(batches):
        assert batch.column_names == ["int", "str"]

    del arrow_table
    del my_table

    reloaded_table.delete_configs()

    reloaded_table.rm()
    assert not reloaded_table.exists_in_system()


def test_create_and_reload_pandas_data_from_s3(pandas_table, table_s3_bucket):
    name = "@/my_test_pandas_table"

    my_table = (
        rh.table(
            data=pandas_table,
            name=name,
            path=f"/{table_s3_bucket}/pandas_df",
            system="s3",
            mkdir=True,
        )
        .write()
        .save()
    )

    reloaded_table = rh.table(name=name)
    reloaded_data: ray.data.Dataset = reloaded_table.data
    assert pandas_table.equals(reloaded_data.to_pandas())

    batches = reloaded_table.stream(batch_size=10)
    for idx, batch in enumerate(batches):
        assert isinstance(batch, pd.DataFrame)
        assert batch["id"].tolist() == list(range(1, 7))

    del pandas_table
    del my_table

    reloaded_table.delete_configs()

    reloaded_table.rm()
    assert not reloaded_table.exists_in_system()


def test_create_and_reload_huggingface_data_from_s3(huggingface_table, table_s3_bucket):
    name = "@/my_test_hf_table"

    my_table = (
        rh.table(
            data=huggingface_table,
            name=name,
            path=f"/{table_s3_bucket}/huggingface_data",
            system="s3",
            mkdir=True,
        )
        .write()
        .save()
    )

    reloaded_table = rh.table(name=name)

    # Stream in as huggingface dataset
    batches = reloaded_table.stream(batch_size=10, as_dict=False)
    for idx, batch in enumerate(batches):
        assert batch.column_names == ["label", "text"]
        assert batch.shape == (10, 2)

    del huggingface_table
    del my_table

    reloaded_table.delete_configs()

    reloaded_table.rm()
    assert not reloaded_table.exists_in_system()


def test_create_and_reload_dask_data_from_s3(dask_table, table_s3_bucket):
    name = "@/my_test_dask_table"

    my_table = (
        rh.table(
            data=dask_table,
            name=name,
            path=f"/{table_s3_bucket}/dask",
            system="s3",
            mkdir=True,
        )
        .write()
        .save()
    )

    reloaded_table = rh.table(name=name)
    reloaded_data: "dask.dataframe.core.DataFrame" = reloaded_table.data.to_dask()
    assert reloaded_data.columns.to_list() == ["a", "b"]

    batches = reloaded_table.stream(batch_size=10)
    for idx, batch in enumerate(batches):
        assert batch.column_names == ["a", "b"]
        assert batch.shape == (10, 2)

    del dask_table
    del my_table

    reloaded_table.delete_configs()

    reloaded_table.rm()
    assert not reloaded_table.exists_in_system()


def test_create_and_reload_ray_data_from_s3(ray_table, table_s3_bucket):
    name = "@/my_test_ray_table"

    my_table = (
        rh.table(
            data=ray_table,
            name=name,
            path=f"/{table_s3_bucket}/ray_data",
            system="s3",
            mkdir=True,
        )
        .write()
        .save()
    )

    reloaded_table = rh.table(name=name)
    reloaded_data: ray.data.Dataset = reloaded_table.data
    assert reloaded_data.to_pandas().equals(ray_table.to_pandas())

    batches = reloaded_table.stream(batch_size=10)
    for idx, batch in enumerate(batches):
        assert isinstance(batch, pa.Table)
        if idx in [0, 10, 33]:
            # Some random batches to check
            assert [isinstance(val, int) for val in batch["value"].to_pylist()]

    del ray_table
    del my_table

    reloaded_table.delete_configs()

    reloaded_table.rm()
    assert not reloaded_table.exists_in_system()


# ----------------- Iter -----------------
def test_load_pandas_data_as_iter(pandas_table, table_s3_bucket):
    name = "@/my_test_pandas_table"

    my_table = (
        rh.table(
            data=pandas_table,
            name=name,
            path=f"/{table_s3_bucket}/pandas",
            system="s3",
            mkdir=True,
        )
        .write()
        .save()
    )

    reloaded_table = rh.table(name=name)
    reloaded_data: ray.data.Dataset = next(iter(reloaded_table))

    assert isinstance(reloaded_data, pd.Series)
    assert reloaded_data.to_dict() == {"id": 1, "grade": "a"}

    del pandas_table
    del my_table

    reloaded_table.delete_configs()

    reloaded_table.rm()
    assert not reloaded_table.exists_in_system()


def test_load_pyarrow_data_as_iter(arrow_table, table_s3_bucket):
    name = "@/my_test_pyarrow_table"

    my_table = (
        rh.table(
            data=arrow_table,
            name=name,
            path=f"/{table_s3_bucket}/pyarrow-data",
            system="s3",
            mkdir=True,
        )
        .write()
        .save()
    )

    reloaded_table = rh.table(name=name)
    reloaded_data: pa.ChunkedArray = next(iter(reloaded_table))

    assert isinstance(reloaded_data, pa.ChunkedArray)
    assert reloaded_data.to_pylist() == list(range(1, 11))

    del arrow_table
    del my_table

    reloaded_table.delete_configs()

    reloaded_table.rm()
    assert not reloaded_table.exists_in_system()


def test_load_huggingface_data_as_iter(huggingface_table, table_s3_bucket):
    name = "@/my_test_huggingface_table"

    my_table = (
        rh.table(
            data=huggingface_table,
            name=name,
            path=f"/{table_s3_bucket}/huggingface-dataset",
            system="s3",
            mkdir=True,
        )
        .write()
        .save()
    )

    reloaded_table = rh.table(name=name)
    reloaded_data: pa.ChunkedArray = next(iter(reloaded_table))
    assert isinstance(reloaded_data, pa.ChunkedArray)

    del huggingface_table
    del my_table

    reloaded_table.delete_configs()

    reloaded_table.rm()
    assert not reloaded_table.exists_in_system()


# ----------------- Shuffling -----------------
def test_shuffling_pyarrow_data_from_s3(arrow_table, table_s3_bucket):
    name = "@/my_test_shuffled_pyarrow_table"

    my_table = (
        rh.table(
            data=arrow_table,
            name=name,
            path=f"/{table_s3_bucket}/pyarrow",
            system="s3",
            mkdir=True,
        )
        .write()
        .save()
    )

    reloaded_table = rh.table(name=name)
    batches = reloaded_table.stream(
        batch_size=10, shuffle_seed=42, shuffle_buffer_size=10
    )
    for idx, batch in enumerate(batches):
        assert isinstance(batch, pa.Table)
        assert arrow_table.columns[0].to_pylist() != batch.columns[0].to_pylist()
        assert arrow_table.columns[1].to_pylist() != batch.columns[1].to_pylist()

    del arrow_table
    del my_table

    reloaded_table.delete_configs()

    reloaded_table.rm()
    assert not reloaded_table.exists_in_system()


# -------------------------------------------------
# ----------------- Cluster tests -----------------
# -------------------------------------------------


def test_create_and_reload_pandas_data_from_cluster(pandas_table, cluster):
    # Make sure the destination folder for the data exists on the cluster
    data_path_on_cluster = f"{Folder.DEFAULT_CACHE_FOLDER}/pandas-data"
    cluster.run([f"mkdir -p {data_path_on_cluster}"])

    name = "@/my_test_pandas_table"
    my_table = (
        rh.table(
            data=pandas_table,
            name=name,
            path=data_path_on_cluster,
            system=cluster,
        )
        .write()
        .save()
    )

    reloaded_table = rh.table(name=name)

    reloaded_data: ray.data.Dataset = reloaded_table.data
    assert pandas_table.equals(reloaded_data.to_pandas())

    batches = reloaded_table.stream(batch_size=10)
    for idx, batch in enumerate(batches):
        assert isinstance(batch, pd.DataFrame)
        assert batch["id"].tolist() == list(range(1, 7))

    del pandas_table
    del my_table

    reloaded_table.delete_configs()

    reloaded_table.rm()
    assert not reloaded_table.exists_in_system()


def test_create_and_reload_ray_data_from_cluster(ray_table, cluster):
    data_path_on_cluster = f"{Folder.DEFAULT_CACHE_FOLDER}/ray-data"
    cluster.run([f"mkdir -p {data_path_on_cluster}"])

    name = "@/my_test_ray_cluster_table"

    my_table = (
        rh.table(
            data=ray_table,
            name=name,
            path=data_path_on_cluster,
            system=cluster,
        )
        .write()
        .save()
    )

    reloaded_table = rh.table(name=name)
    reloaded_data: ray.data.Dataset = reloaded_table.data
    assert ray_table.to_pandas().equals(reloaded_data.to_pandas())

    batches = reloaded_table.stream(batch_size=10)
    for idx, batch in enumerate(batches):
        assert isinstance(batch, pa.Table)
        if idx in [0, 10, 33]:
            # Some random batches to check
            assert [isinstance(val, int) for val in batch["value"].to_pylist()]

    del ray_table
    del my_table

    reloaded_table.delete_configs()

    reloaded_table.rm()
    assert not reloaded_table.exists_in_system()


def test_create_and_reload_pyarrow_data_from_cluster(arrow_table, cluster):
    data_path_on_cluster = f"{Folder.DEFAULT_CACHE_FOLDER}/pyarrow-data"
    cluster.run([f"mkdir -p {data_path_on_cluster}"])

    name = "@/my_test_pyarrow_cluster_table"

    my_table = (
        rh.table(
            data=arrow_table,
            name=name,
            path=data_path_on_cluster,
            system=cluster,
        )
        .write()
        .save()
    )

    reloaded_table = rh.table(name=name)
    reloaded_data: ray.data.Dataset = reloaded_table.data
    assert arrow_table.to_pandas().equals(reloaded_data.to_pandas())

    batches = reloaded_table.stream(batch_size=10)
    for idx, batch in enumerate(batches):
        assert batch.column_names == ["int", "str"]

    del arrow_table
    del my_table

    reloaded_table.delete_configs()

    reloaded_table.rm()
    assert not reloaded_table.exists_in_system()


def test_create_and_reload_huggingface_data_from_cluster(huggingface_table, cluster):
    data_path_on_cluster = f"{Folder.DEFAULT_CACHE_FOLDER}/hf-data"
    cluster.run([f"mkdir -p {data_path_on_cluster}"])

    name = "@/my_test_hf_cluster_table"

    my_table = (
        rh.table(
            data=huggingface_table,
            name=name,
            path=data_path_on_cluster,
            system=cluster,
        )
        .write()
        .save()
    )

    reloaded_table = rh.table(name=name)
    reloaded_data: ray.data.Dataset = reloaded_table.data
    assert huggingface_table.to_pandas().equals(reloaded_data.to_pandas())

    # Stream in as huggingface dataset
    batches = reloaded_table.stream(batch_size=10, as_dict=False)
    for idx, batch in enumerate(batches):
        assert batch.column_names == ["label", "text"]
        assert batch.shape == (10, 2)

    del huggingface_table
    del my_table

    reloaded_table.delete_configs()

    reloaded_table.rm()
    assert not reloaded_table.exists_in_system()


def test_create_and_reload_dask_data_from_cluster(dask_table, cluster):
    data_path_on_cluster = f"{Folder.DEFAULT_CACHE_FOLDER}/dask-data"
    cluster.run([f"mkdir -p {data_path_on_cluster}"])

    name = "@/my_test_dask_cluster_table"

    my_table = (
        rh.table(
            data=dask_table,
            name=name,
            path=data_path_on_cluster,
            system=cluster,
        )
        .write()
        .save()
    )

    reloaded_table = rh.table(name=name)
    reloaded_data: "dask.dataframe.core.DataFrame" = reloaded_table.data.to_dask()
    assert reloaded_data.columns.to_list() == ["a", "b"]

    batches = reloaded_table.stream(batch_size=10)
    for idx, batch in enumerate(batches):
        assert batch.column_names == ["a", "b"]
        assert batch.shape == (10, 2)

    del dask_table
    del my_table

    reloaded_table.delete_configs()

    reloaded_table.rm()
    assert not reloaded_table.exists_in_system()


def test_to_cluster_attr(pandas_table, cluster, tmp_path):
    local_path = tmp_path / "table_tests/local_test_table"
    name = "~/my_local_test_table"

    my_table = (
        rh.table(
            data=pandas_table,
            name=name,
            path=str(local_path),
            system="file",
        )
        .write()
        .save()
    )

    cluster_table = my_table.to(system=cluster)

    assert isinstance(cluster_table.system, rh.Cluster)
    assert cluster_table._folder._fs_str == "ssh"

    data = cluster_table.data
    assert data.to_pandas().equals(pandas_table)

    del pandas_table
    del my_table

    cluster_table.delete_configs()
    cluster_table.rm()
    assert not cluster_table.exists_in_system()


# -------------------------------------------------
# ----------------- Fetching tests -----------------
# -------------------------------------------------
def test_create_and_fetch_pyarrow_data_from_s3(arrow_table, table_s3_bucket):
    name = "@/my_test_fetch_pyarrow_table"

    my_table = (
        rh.table(
            data=arrow_table,
            name=name,
            path=f"/{table_s3_bucket}/pyarrow",
            system="s3",
            mkdir=True,
        )
        .write()
        .save()
    )

    reloaded_table = rh.table(name=name)
    reloaded_data: pa.Table = reloaded_table.fetch()
    assert arrow_table == reloaded_data

    del arrow_table
    del my_table

    reloaded_table.delete_configs()

    reloaded_table.rm()
    assert not reloaded_table.exists_in_system()


def test_create_and_fetch_pandas_data_from_s3(pandas_table, table_s3_bucket):
    name = "@/my_test_fetch_pandas_table"

    my_table = (
        rh.table(
            data=pandas_table,
            name=name,
            path=f"/{table_s3_bucket}/pandas",
            system="s3",
            mkdir=True,
        )
        .write()
        .save()
    )

    reloaded_table = rh.table(name=name)
    reloaded_data: pd.DataFrame = reloaded_table.fetch()
    assert pandas_table.equals(reloaded_data)

    del pandas_table
    del my_table

    reloaded_table.delete_configs()

    reloaded_table.rm()
    assert not reloaded_table.exists_in_system()


def test_create_and_fetch_huggingface_data_from_s3(huggingface_table, table_s3_bucket):
    name = "@/my_test_fetch_huggingface_table"

    my_table = (
        rh.table(
            data=huggingface_table,
            name=name,
            path=f"/{table_s3_bucket}/huggingface",
            system="s3",
            mkdir=True,
        )
        .write()
        .save()
    )

    reloaded_table = rh.table(name=name)
    reloaded_data = reloaded_table.fetch()
    assert huggingface_table.shape == reloaded_data.shape

    del huggingface_table
    del my_table

    reloaded_table.delete_configs()

    reloaded_table.rm()
    assert not reloaded_table.exists_in_system()


def test_create_and_fetch_ray_data_from_s3(ray_table, table_s3_bucket):
    name = "@/my_test_fetch_ray_table"

    my_table = (
        rh.table(
            data=ray_table,
            name=name,
            path=f"/{table_s3_bucket}/ray",
            system="s3",
            mkdir=True,
        )
        .write()
        .save()
    )

    reloaded_table = rh.table(name=name)
    reloaded_data: ray.data.Dataset = reloaded_table.fetch()
    assert ray_table.to_pandas().equals(reloaded_data.to_pandas())

    del ray_table
    del my_table

    reloaded_table.delete_configs()

    reloaded_table.rm()
    assert not reloaded_table.exists_in_system()


def test_create_and_fetch_dask_data_from_s3(dask_table, table_s3_bucket):
    name = "@/my_test_fetch_dask_table"

    my_table = (
        rh.table(
            data=dask_table,
            name=name,
            path=f"/{table_s3_bucket}/dask",
            system="s3",
            mkdir=True,
        )
        .write()
        .save()
    )

    reloaded_table = rh.table(name=name)
    reloaded_data: "dask.dataframe.core.DataFrame" = reloaded_table.fetch()
    assert dask_table.npartitions == reloaded_data.npartitions

    del dask_table
    del my_table

    reloaded_table.delete_configs()

    reloaded_table.rm()
    assert not reloaded_table.exists_in_system()


# -------------------------------------------------
# ----------------- Table Sharing tests -----------------
# -------------------------------------------------
def test_sharing_table(pandas_table):
    name = "shared_pandas_table"

    my_table = (
        rh.table(
            data=pandas_table,
            name=name,
            system="s3",
            mkdir=True,
        )
        .write()
        .save()
    )

    my_table.share(
        users=["donny@run.house", "josh@run.house"],
        access_level="write",
        notify_users=False,
    )

    assert my_table.exists_in_system()


def test_read_shared_table():
    my_table = rh.table(name="@/shared_pandas_table")
    df: pd.DataFrame = my_table.fetch()
    assert not df.empty
