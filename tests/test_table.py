import unittest
from pathlib import Path

import datasets
import pyarrow as pa
import pandas as pd
import ray.data

import runhouse as rh

TEMP_LOCAL_FOLDER = Path("~/.rh/temp").expanduser()
BUCKET_NAME = 'runhouse-tests'
NUM_PARTITIONS = 10


def setup():
    # Create buckets in S3
    from sky.data.storage import S3Store
    S3Store(name=BUCKET_NAME, source='')


def tokenize_function(examples):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
    return tokenizer(examples["text"], padding="max_length", truncation=True)


def load_sample_data(data_type):
    if data_type == 'huggingface':
        from datasets import load_dataset
        dataset = load_dataset("yelp_review_full", split='train[:10%]')
        return dataset

    elif data_type == 'pyarrow':
        df = pd.DataFrame({'int': [1, 2], 'str': ['a', 'b']})
        arrow_table = pa.Table.from_pandas(df)
        return arrow_table

    elif data_type == 'cudf':
        import cudf
        gdf = cudf.DataFrame({"id": [1, 2, 3, 4, 5, 6], "grade": ['a', 'b', 'b', 'a', 'a', 'e']})
        return gdf

    elif data_type == 'pandas':
        df = pd.DataFrame({"id": [1, 2, 3, 4, 5, 6], "grade": ['a', 'b', 'b', 'a', 'a', 'e']})
        return df

    elif data_type == 'dask':
        import dask.dataframe as dd
        index = pd.date_range("2021-09-01", periods=2400, freq="1H")
        df = pd.DataFrame({"a": range(2400), "b": list("abcaddbe" * 300)}, index=index)
        ddf = dd.from_pandas(df, npartitions=NUM_PARTITIONS)
        return ddf

    elif data_type == 'ray':
        import ray
        ds = ray.data.range(10000)
        return ds

    else:
        raise Exception(f"Unsupported data type {data_type}")


# ----------------- Run tests -----------------

def test_create_and_reload_pandas_locally():
    orig_data = load_sample_data('pandas')

    my_table = rh.table(data=orig_data,
                        name='~/my_test_pandas_table',
                        url='table_tests/test_pandas_table',
                        fs='file',
                        mkdir=True).save()

    reloaded_table = rh.table(name='~/my_test_pandas_table', dryrun=True)
    reloaded_data: ray.data.Dataset = reloaded_table.data

    assert orig_data.equals(reloaded_data.to_pandas())

    reloaded_table.delete_configs()

    reloaded_table.delete_in_fs()
    assert not reloaded_table.exists_in_fs()


def test_create_and_reload_pyarrow_locally():
    orig_data = load_sample_data('pyarrow')
    my_table = rh.table(data=orig_data,
                        name='~/my_test_pyarrow_table',
                        url='table_tests/pyarrow_test_table',
                        fs='file',
                        mkdir=True).save()

    reloaded_table = rh.table(name='~/my_test_pyarrow_table', dryrun=True)
    reloaded_data: ray.data.Dataset = reloaded_table.data

    assert orig_data.to_pandas().equals(reloaded_data.to_pandas())

    del orig_data
    del my_table

    reloaded_table.delete_configs()

    reloaded_table.delete_in_fs()
    assert not reloaded_table.exists_in_fs()


def test_create_and_reload_dask_data_from_s3():
    orig_data = load_sample_data(data_type='dask')
    my_table = rh.table(data=orig_data,
                        name='@/my_test_dask_table',
                        url=f'/{BUCKET_NAME}/dask',
                        fs='s3',
                        mkdir=True).save()

    reloaded_table = rh.table(name='@/my_test_dask_table', dryrun=True)
    reloaded_data: ray.data.Dataset = reloaded_table.data
    dask_reloaded_data = reloaded_data.to_dask()

    assert orig_data.to_pandas().equals(dask_reloaded_data.to_pandas())

    del orig_data
    del my_table

    reloaded_table.delete_configs()

    reloaded_table.delete_in_fs()
    assert not reloaded_table.exists_in_fs()


def test_create_and_reload_ray_data_from_s3():
    orig_data = load_sample_data(data_type='ray')

    my_table = rh.table(data=orig_data,
                        name='@/my_test_ray_table',
                        url=f'/{BUCKET_NAME}/ray',
                        fs='s3',
                        mkdir=True).save()

    reloaded_table = rh.table(name='@/my_test_ray_table', dryrun=True)
    reloaded_data: ray.data.Dataset = reloaded_table.data
    assert reloaded_data.to_pandas().equals(orig_data.to_pandas())

    del orig_data
    del my_table

    reloaded_table.delete_configs()

    reloaded_table.delete_in_fs()
    assert not reloaded_table.exists_in_fs()


def test_create_and_reload_pyarrow_data_from_s3():
    orig_data = load_sample_data(data_type='pyarrow')

    my_table = rh.table(data=orig_data,
                        name='@/my_test_pyarrow_table',
                        url=f'/{BUCKET_NAME}/pyarrow',
                        fs='s3',
                        mkdir=True).save()

    reloaded_table = rh.table(name='@/my_test_pyarrow_table', dryrun=True)
    reloaded_data: ray.data.Dataset = reloaded_table.data
    assert reloaded_data.to_pandas().equals(orig_data.to_pandas())

    del orig_data

    reloaded_table.delete_configs()

    reloaded_table.delete_in_fs()
    assert not reloaded_table.exists_in_fs()


def test_create_and_reload_pandas_data_from_s3():
    orig_data = load_sample_data(data_type='pandas')

    my_table = rh.table(data=orig_data,
                        name='@/my_test_pandas_table',
                        url=f'/{BUCKET_NAME}/pandas_df',
                        fs='s3',
                        mkdir=True).save()

    reloaded_table = rh.table(name='@/my_test_pandas_table', dryrun=True)
    reloaded_data: ray.data.Dataset = reloaded_table.data
    assert orig_data.equals(reloaded_data.to_pandas())

    del orig_data
    del my_table

    reloaded_table.delete_configs()

    reloaded_table.delete_in_fs()
    assert not reloaded_table.exists_in_fs()


def test_create_and_reload_huggingface_data_from_s3():
    orig_data: datasets.Dataset = load_sample_data(data_type='huggingface')

    my_table = rh.table(data=orig_data,
                        name='@/my_test_hf_table',
                        url=f'/{BUCKET_NAME}/huggingface',
                        fs='s3',
                        mkdir=True).save()

    reloaded_table = rh.table(name='@/my_test_hf_table', dryrun=True)
    reloaded_data: ray.data.Dataset = reloaded_table.data
    assert reloaded_data.to_pandas().equals(orig_data.to_pandas())

    del orig_data
    del my_table

    reloaded_table.delete_configs()

    reloaded_table.delete_in_fs()
    assert not reloaded_table.exists_in_fs()


def test_create_and_stream_huggingface_data_from_s3():
    orig_data: datasets.Dataset = load_sample_data(data_type='huggingface')

    my_table = rh.table(data=orig_data,
                        name='@/my_test_hf_stream_table',
                        url=f'/{BUCKET_NAME}/huggingface-stream',
                        fs='s3',
                        mkdir=True).save()

    reloaded_table = rh.table(name='@/my_test_hf_stream_table', dryrun=True)

    # Stream in as huggingface dataset
    batches = reloaded_table.stream(batch_size=10, as_dict=False)
    for idx, batch in enumerate(batches):
        assert batch.column_names == ['label', 'text']
        assert batch.shape == (10, 2)

    del orig_data
    del my_table

    reloaded_table.delete_configs()

    reloaded_table.delete_in_fs()
    assert not reloaded_table.exists_in_fs()


def test_shuffling_data():
    # [TODO] JL
    pass


def test_load_pandas_data_as_iter():
    orig_data = load_sample_data(data_type='pandas')

    my_table = rh.table(data=orig_data,
                        name='@/my_test_pandas_table',
                        url=f'/{BUCKET_NAME}/pandas',
                        fs='s3',
                        mkdir=True).save()

    reloaded_table = rh.table(name='@/my_test_pandas_table', dryrun=True)
    reloaded_data: ray.data.Dataset = next(iter(reloaded_table))
    assert isinstance(reloaded_data, pd.Series)

    del orig_data
    del my_table

    reloaded_table.delete_configs()

    reloaded_table.delete_in_fs()
    assert not reloaded_table.exists_in_fs()


def test_load_pyarrow_data_as_iter():
    orig_data = load_sample_data(data_type='pyarrow')

    my_table = rh.table(data=orig_data,
                        name='@/my_test_pyarrow_table',
                        url=f'/{BUCKET_NAME}/pyarrow',
                        fs='s3',
                        mkdir=True).save()

    reloaded_table = rh.table(name='@/my_test_pyarrow_table', dryrun=True)
    reloaded_data: pa.ChunkedArray = next(iter(reloaded_table))
    assert isinstance(reloaded_data, pa.ChunkedArray)

    del orig_data
    del my_table

    reloaded_table.delete_configs()

    reloaded_table.delete_in_fs()
    assert not reloaded_table.exists_in_fs()


def test_load_huggingface_data_as_iter():
    orig_data = load_sample_data(data_type='huggingface')

    my_table = rh.table(data=orig_data,
                        name='@/my_test_huggingface_table',
                        url=f'/{BUCKET_NAME}/huggingface',
                        fs='s3',
                        mkdir=True).save()

    reloaded_table = rh.table(name='@/my_test_huggingface_table', dryrun=True)
    reloaded_data: pa.ChunkedArray = next(iter(reloaded_table))
    assert isinstance(reloaded_data, pa.ChunkedArray)

    del orig_data
    del my_table

    reloaded_table.delete_configs()

    reloaded_table.delete_in_fs()
    assert not reloaded_table.exists_in_fs()


def test_create_and_reload_partitioned_data_from_s3():
    # TODO [JL] partitioning currently only implemented with pyarrow API - see if we can do this with ray
    data = load_sample_data("pyarrow")

    my_table = rh.table(data=data,
                        name='@/partitioned_my_test_table',
                        url=f'/{BUCKET_NAME}/pyarrow-partitioned',
                        partition_cols=['int'],
                        fs='s3',
                        mkdir=True).save()

    reloaded_table = rh.table(name='@/partitioned_my_test_table', dryrun=True)

    # Let's reload only the column we partitioned on
    reloaded_data = reloaded_table.fetch(columns=['int'])
    assert reloaded_data.shape == (2, 1)

    del data
    del my_table

    reloaded_table.delete_configs()

    reloaded_table.delete_in_fs()
    assert not reloaded_table.exists_in_fs()


def test_stream_data_from_file():
    from pathlib import Path
    url = 'table_tests/stream_data'
    Path(Path.cwd().parent / url).mkdir(parents=True, exist_ok=True)

    data = pd.DataFrame({'my_col': list(range(50))})
    my_table = rh.table(data=data,
                        name='~/my_test_table',
                        url=url,
                        fs='file').save()

    reloaded_table = rh.table(name='~/my_test_table', dryrun=True)

    batches = reloaded_table.stream(batch_size=10)
    for idx, batch in enumerate(batches):
        assert isinstance(batch, pd.DataFrame)
        assert batch['my_col'].tolist() == list(range(idx * 10, (idx + 1) * 10))

    reloaded_table.delete_configs()

    reloaded_table.delete_in_fs()
    assert not reloaded_table.exists_in_fs()


def test_stream_data_from_s3():
    data = load_sample_data('pyarrow')
    my_table = rh.table(data=data,
                        name='@/my_test_table',
                        url=f'/{BUCKET_NAME}/stream-data',
                        fs='s3',
                        mkdir=True).save()

    reloaded_table = rh.table(name='@/my_test_table', dryrun=True)

    batches = reloaded_table.stream(batch_size=10)
    for idx, batch in enumerate(batches):
        assert batch.column_names == ['int', 'str']

    reloaded_table.delete_configs()

    reloaded_table.delete_in_fs()
    assert not reloaded_table.exists_in_fs()


# TODO [JL] Add more tests where data source is SSH (i.e. data lives on a cluster)
# TODO [JL] Read / write / streaming works for each location

def test_create_and_reload_pandas_data_from_cluster():
    from runhouse import Folder
    c1 = rh.cluster(name='^rh-cpu').up_if_not()

    # Make sure the destination folder for the data exists on the cluster
    data_url_on_cluster = Folder.DEFAULT_CACHE_FOLDER
    c1.run([f'mkdir -p {data_url_on_cluster}'])

    orig_data = load_sample_data(data_type='pandas')

    my_table = rh.table(data=orig_data,
                        name='@/my_test_pandas_table',
                        url=data_url_on_cluster,
                        fs=c1).save()

    reloaded_table = rh.table(name='@/my_test_pandas_table', dryrun=True)

    reloaded_data: ray.data.Dataset = reloaded_table.data
    assert orig_data.equals(reloaded_data.to_pandas())

    del orig_data
    del my_table

    reloaded_table.delete_configs()

    reloaded_table.delete_in_fs()
    assert not reloaded_table.exists_in_fs()


def test_create_and_reload_ray_data_from_cluster():
    from runhouse import Folder
    c1 = rh.cluster('^rh-cpu').up_if_not()

    # Make sure the destination folder for the data exists on the cluster. Here we'll create a separate folder since
    # ray will split the data into multiple parquet files for us
    data_url_on_cluster = f'{Folder.DEFAULT_CACHE_FOLDER}/ray-data'
    c1.run([f'mkdir -p {data_url_on_cluster}'])

    orig_data = load_sample_data(data_type='ray')

    my_table = rh.table(data=orig_data,
                        name='@/my_test_ray_cluster_table',
                        url=data_url_on_cluster,
                        fs=c1).save()

    reloaded_table = rh.table(name='@/my_test_ray_cluster_table', dryrun=True)
    reloaded_data: ray.data.Dataset = reloaded_table.data
    assert orig_data.to_pandas().equals(reloaded_data.to_pandas())

    del orig_data

    reloaded_table.delete_configs()

    reloaded_table.delete_in_fs()
    assert not reloaded_table.exists_in_fs()


def test_create_and_reload_pyarrow_data_from_cluster():
    from runhouse import Folder
    c1 = rh.cluster('^rh-cpu').up_if_not()

    # Make sure the destination folder for the data exists on the cluster. Here we'll create a separate folder since
    # ray will split the data into multiple parquet files for us
    data_url_on_cluster = f'{Folder.DEFAULT_CACHE_FOLDER}/pyarrow-data'
    c1.run([f'mkdir -p {data_url_on_cluster}'])

    orig_data = load_sample_data(data_type='pyarrow')

    my_table = rh.table(data=orig_data,
                        name='@/my_test_pyarrow_cluster_table',
                        url=data_url_on_cluster,
                        fs=c1).save()

    reloaded_table = rh.table(name='@/my_test_pyarrow_cluster_table', dryrun=True)
    reloaded_data: ray.data.Dataset = reloaded_table.data
    assert orig_data.to_pandas().equals(reloaded_data.to_pandas())

    del orig_data

    reloaded_table.delete_configs()

    reloaded_table.delete_in_fs()
    assert not reloaded_table.exists_in_fs()


def test_create_and_reload_huggingface_data_from_cluster():
    from runhouse import Folder
    c1 = rh.cluster('^rh-cpu').up_if_not()

    # Make sure the destination folder for the data exists on the cluster
    data_url_on_cluster = f'{Folder.DEFAULT_CACHE_FOLDER}/hf-data'
    c1.run([f'mkdir -p {data_url_on_cluster}'])

    orig_data = load_sample_data(data_type='huggingface')

    my_table = rh.table(data=orig_data,
                        name='@/my_test_hf_cluster_table',
                        url=data_url_on_cluster,
                        fs=c1).save()

    reloaded_table = rh.table(name='@/my_test_hf_cluster_table', dryrun=True)
    reloaded_data: ray.data.Dataset = reloaded_table.data
    assert orig_data.to_pandas().equals(reloaded_data.to_pandas())

    del orig_data

    reloaded_table.delete_configs()

    reloaded_table.delete_in_fs()
    assert not reloaded_table.exists_in_fs()


if __name__ == '__main__':
    setup()
    unittest.main()
