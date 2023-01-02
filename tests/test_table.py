import unittest
from pathlib import Path

import datasets
import pyarrow as pa
import pandas as pd

import runhouse as rh

TEMP_LOCAL_FOLDER = Path("~/.rh/temp").expanduser()
BUCKET_NAME = 'runhouse-tests'
NUM_PARTITIONS = 10


def setup():
    import boto3
    s3 = boto3.client('s3')
    s3.create_bucket(Bucket=BUCKET_NAME)


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
        pa_table = pa.Table.from_pandas(df)
        return pa_table

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

def test_create_and_reload_pandas_from_file():
    data = load_sample_data('pandas')
    orig_data_shape = data.shape
    my_table = rh.table(data=data,
                        name='my_test_pandas_table',
                        url='table_tests/test_pandas_table.parquet',
                        save_to=['local'],
                        fs='file',
                        mkdir=True)

    reloaded_table = rh.table(name='my_test_pandas_table', load_from=['local'], dryrun=True)
    reloaded_data: pd.DataFrame = reloaded_table.data

    assert reloaded_data.shape == orig_data_shape

    del data
    del my_table

    reloaded_table.delete_in_fs()
    assert not reloaded_table.exists_in_fs()

    reloaded_table.delete_configs(delete_from=['local'])


def test_create_and_reload_pyarrow_from_file():
    data = load_sample_data('pyarrow')
    orig_data_shape = data.shape
    my_table = rh.table(data=data,
                        name='my_test_pyarrow_table',
                        url='table_tests/pyarrow_test_table',
                        save_to=['local'],
                        fs='file',
                        mkdir=True)

    reloaded_table = rh.table(name='my_test_pyarrow_table', load_from=['local'], dryrun=True)
    reloaded_data: pd.DataFrame = reloaded_table.data

    assert reloaded_data.shape == orig_data_shape

    del data
    del my_table

    reloaded_table.delete_in_fs()
    assert not reloaded_table.exists_in_fs()

    reloaded_table.delete_configs(delete_from=['local'])


def test_create_and_reload_dask_data_from_s3():
    import dask
    orig_data = load_sample_data(data_type='dask')
    my_table = rh.table(data=orig_data,
                        name='my_test_dask_table',
                        url=f'{BUCKET_NAME}/dask',
                        save_to=['rns'],
                        fs='s3',
                        mkdir=True)

    reloaded_table = rh.table(name='my_test_dask_table', load_from=['rns'], dryrun=True)
    reloaded_data = reloaded_table.data
    assert isinstance(reloaded_data, dask.dataframe.core.DataFrame)

    del orig_data
    del my_table

    reloaded_table.delete_configs(delete_from=['rns'])

    reloaded_table.delete_in_fs()
    assert not reloaded_table.exists_in_fs()


def test_create_and_reload_ray_data_from_s3():
    orig_data = load_sample_data(data_type='ray')

    my_table = rh.table(data=orig_data,
                        name='my_test_ray_table',
                        url=f'{BUCKET_NAME}/ray',
                        save_to=['rns'],
                        fs='s3',
                        mkdir=True)

    reloaded_table = rh.table(name='my_test_ray_table', load_from=['rns'], dryrun=True)
    reloaded_data = reloaded_table.data
    assert reloaded_data

    del orig_data
    del my_table

    reloaded_table.delete_configs(delete_from=['rns'])

    reloaded_table.delete_in_fs()
    assert not reloaded_table.exists_in_fs()


def test_create_and_reload_pyarrow_data_from_s3():
    orig_data = load_sample_data(data_type='pyarrow')

    my_table = rh.table(data=orig_data,
                        name='my_test_pyarrow_table',
                        url=f'{BUCKET_NAME}/pyarrow',
                        save_to=['rns'],
                        fs='s3',
                        mkdir=True)

    reloaded_table = rh.table(name='my_test_pyarrow_table', load_from=['rns'], dryrun=True)
    reloaded_data = reloaded_table.data
    assert reloaded_data

    del orig_data
    del my_table

    reloaded_table.delete_configs(delete_from=['rns'])

    reloaded_table.delete_in_fs()
    assert not reloaded_table.exists_in_fs()


def test_create_and_reload_pandas_data_from_s3():
    orig_data = load_sample_data(data_type='pandas')

    my_table = rh.table(data=orig_data,
                        name='my_test_pandas_table',
                        url=f'{BUCKET_NAME}/pandas_df.parquet',
                        save_to=['rns'],
                        fs='s3',
                        mkdir=True)

    reloaded_table = rh.table(name='my_test_pandas_table', load_from=['rns'], dryrun=True)
    reloaded_data: pd.DataFrame = reloaded_table.data
    assert orig_data.equals(reloaded_data)

    del orig_data
    del my_table

    reloaded_table.delete_configs(delete_from=['rns'])

    reloaded_table.delete_in_fs()
    assert not reloaded_table.exists_in_fs()


def test_create_and_reload_huggingface_data_from_s3():
    orig_data: datasets.arrow_dataset.Dataset = load_sample_data(data_type='huggingface')
    orig_data_shape = orig_data.shape

    # Convert data to a type Runhouse can work with (pyarrow or pandas)
    # Note: You can also convert the table to pandas using: `orig_data.to_pandas()`
    orig_data_pa: pa.Table = orig_data.data.table

    my_table = rh.table(data=orig_data_pa,
                        name='my_test_hf_table',
                        url=f'{BUCKET_NAME}/huggingface',
                        save_to=['rns'],
                        fs='s3',
                        mkdir=True)

    reloaded_table = rh.table(name='my_test_hf_table', load_from=['rns'], dryrun=True)
    reloaded_data = reloaded_table.data
    assert reloaded_data.shape == orig_data_shape

    del orig_data
    del my_table

    reloaded_table.delete_configs(delete_from=['rns'])

    reloaded_table.delete_in_fs()
    assert not reloaded_table.exists_in_fs()


def test_create_and_stream_huggingface_data_from_s3():
    orig_data: datasets.arrow_dataset.Dataset = load_sample_data(data_type='huggingface')
    orig_data_shape = orig_data.shape

    # Convert data to a type Runhouse can work with (pyarrow or pandas)
    # Note: You can also convert the table to pandas using: `orig_data.to_pandas()`
    orig_data_pa: pa.Table = orig_data.data.table

    my_table = rh.table(data=orig_data_pa,
                        name='my_test_hf_stream_table',
                        url=f'{BUCKET_NAME}/huggingface-stream',
                        save_to=['rns'],
                        fs='s3',
                        mkdir=True)

    reloaded_table = rh.table(name='my_test_hf_stream_table', load_from=['rns'], dryrun=True)
    reloaded_data = reloaded_table.data
    assert reloaded_data.shape == orig_data_shape

    batches = reloaded_table.stream(batch_size=10)
    for idx, batch in enumerate(batches):
        assert batch.shape == (10, 2)
        assert batch.column_names == ['label', 'text']

    del orig_data
    del my_table

    reloaded_table.delete_configs(delete_from=['rns'])

    reloaded_table.delete_in_fs()
    assert not reloaded_table.exists_in_fs()


def test_create_and_reload_partitioned_data_from_s3():
    data = load_sample_data("pyarrow")
    orig_data_shape = data.shape

    my_table = rh.table(data=data,
                        name='partitioned_my_test_table',
                        url=f'{BUCKET_NAME}/pyarrow-partitioned',
                        partition_cols=['int'],
                        fs='s3',
                        save_to=['rns'],
                        mkdir=True)

    reloaded_table = rh.table(name='partitioned_my_test_table', load_from=['rns'], dryrun=True)

    # Let's reload only the column we partitioned on
    reloaded_data = reloaded_table.fetch(columns=['int'])
    assert reloaded_data.shape == (2, 1)

    del data
    del my_table

    reloaded_table.delete_configs(delete_from=['rns'])

    reloaded_table.delete_in_fs()
    assert not reloaded_table.exists_in_fs()


def test_stream_data_from_file():
    data = pd.DataFrame({'my_col': list(range(50))})
    my_table = rh.table(data=data,
                        name='my_test_table',
                        url='table_tests/stream_data',
                        save_to=['local'],
                        fs='file')

    batches = my_table.stream(batch_size=10)
    for idx, batch in enumerate(batches):
        assert batch['my_col'].to_pylist() == list(range(idx * 10, (idx + 1) * 10))

    my_table.delete_configs(delete_from=['local'])

    my_table.delete_in_fs()
    assert not my_table.exists_in_fs()


def test_stream_data_from_s3():
    data = load_sample_data('pyarrow')
    my_table = rh.table(data=data,
                        name='my_test_table',
                        url=f'{BUCKET_NAME}/stream-data',
                        save_to=['rns'],
                        fs='s3',
                        mkdir=True)

    batches = my_table.stream(batch_size=10)
    for idx, batch in enumerate(batches):
        assert batch.column_names == ['int', 'str']

    my_table.delete_configs(delete_from=['rns'])

    my_table.delete_in_fs()
    assert not my_table.exists_in_fs()


if __name__ == '__main__':
    setup()
    unittest.main()
