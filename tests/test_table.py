import unittest
from pathlib import Path

import pyarrow as pa
import pandas as pd

import runhouse as rh

TEMP_LOCAL_FOLDER = Path("~/.rh/temp").expanduser()
TABLE_NAME = "test_data"
BUCKET_NAME = 'runhouse-tests'

DATA_URL = f"{BUCKET_NAME}/{TABLE_NAME}.parquet"
TEMP_LOCAL_FILE_PATH = str(Path(TEMP_LOCAL_FOLDER / DATA_URL))

S3_ROOT_PATH = f's3://{BUCKET_NAME}'
S3_BUCKET_PATH = f'{BUCKET_NAME}/{TABLE_NAME}'

DATA_TYPES = ['huggingface', 'cudf', 'dask']


def setup():
    import boto3
    s3 = boto3.client('s3')
    s3.put_object(Bucket=BUCKET_NAME, Key=(TABLE_NAME + '/'))


def tokenize_function(examples):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
    return tokenizer(examples["text"], padding="max_length", truncation=True)


def load_sample_data(data_type='huggingface'):
    if data_type == 'huggingface':
        from datasets import load_dataset
        dataset = load_dataset("yelp_review_full")
        tokenized_datasets = dataset.map(tokenize_function, batched=True)
        small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
        return small_train_dataset

    elif data_type == 'cudf':
        import cudf
        gdf = cudf.DataFrame({"id": [1, 2, 3, 4, 5, 6], "grade": ['a', 'b', 'b', 'a', 'a', 'e']})
        return gdf

    elif data_type == 'dask':
        import dask.dataframe as dd
        index = pd.date_range("2021-09-01", periods=2400, freq="1H")
        df = pd.DataFrame({"a": range(2400), "b": list("abcaddbe" * 300)}, index=index)
        ddf = dd.from_pandas(df, npartitions=10)
        return ddf

    else:
        raise Exception(f"Unsupported data type {data_type}")


# ----------------- Run tests -----------------

def test_create_and_reload_from_file():
    data = load_sample_data()
    orig_data_shape = data.shape
    my_table = rh.table(data=data,
                        name='my_test_table',
                        data_url=TEMP_LOCAL_FILE_PATH,
                        data_source='file')

    del data
    del my_table

    reloaded_table = rh.table(name='my_test_table', dryrun=False, load_from=['local'])
    reloaded_data = reloaded_table.data
    assert reloaded_data.to_pandas().shape == orig_data_shape

    reloaded_table.delete_in_fs()
    assert not reloaded_table.folder.exists_in_fs()


def test_create_and_reload_dask_data_from_s3():
    orig_data = load_sample_data(data_type='dask')

    my_table = rh.table(data=orig_data,
                        name='my_test_dask_table',
                        data_url=S3_BUCKET_PATH,
                        data_source='s3')

    del orig_data
    del my_table

    reloaded_table = rh.table(name='my_test_dask_table', dryrun=False, load_from=['rns'])
    reloaded_data: pa.Table = reloaded_table.data
    assert reloaded_data

    reloaded_table.delete_in_fs()
    assert not reloaded_table.folder.exists_in_fs()


def test_create_and_reload_huggingface_data_from_s3():
    orig_data = load_sample_data(data_type='huggingface')
    orig_shape = orig_data.shape

    my_table = rh.table(data=orig_data,
                        name='my_test_hf_table',
                        data_url=S3_BUCKET_PATH,
                        data_source='s3')

    del orig_data
    del my_table

    reloaded_table = rh.table(name='my_test_hf_table', dryrun=False, load_from=['rns'])
    reloaded_data: pa.Table = reloaded_table.data
    assert reloaded_data.shape == orig_shape

    reloaded_table.delete_in_fs()
    assert not reloaded_table.folder.exists_in_fs()


def test_create_and_reload_partitioned_data_from_s3():
    data = load_sample_data("huggingface")

    my_table = rh.table(data=data,
                        name='partitioned_my_test_table',
                        data_url=S3_BUCKET_PATH,
                        data_source='s3',
                        partition_cols=['label'])

    del data
    del my_table

    reloaded_table = rh.table(name='partitioned_my_test_table', dryrun=False, load_from=['rns'])
    reloaded_data: pa.Table = reloaded_table.data
    assert reloaded_data

    reloaded_table.delete_in_fs()
    assert not reloaded_table.folder.exists_in_fs()


def test_stream_data_from_file():
    data = pd.DataFrame({'my_col': list(range(50))})
    my_table = rh.table(data=data,
                        name='my_test_table',
                        data_url=TEMP_LOCAL_FILE_PATH,
                        data_source='file')

    batches = my_table.stream(batch_size=10)
    for idx, batch in enumerate(batches):
        assert batch['my_col'].to_pylist() == list(range(idx * 10, (idx + 1) * 10))


def test_stream_data_from_s3():
    data = load_sample_data()
    my_table = rh.table(data=data,
                        name='my_test_table',
                        data_url=S3_BUCKET_PATH,
                        data_source='s3')

    batches = my_table.stream(batch_size=10)
    for idx, batch in enumerate(batches):
        assert batch.column_names == ['label', 'text', 'input_ids', 'token_type_ids', 'attention_mask']

    my_table.delete_in_fs()
    assert not my_table.folder.exists_in_fs()


def test_create_and_reload_s3():
    data = pd.DataFrame({'my_col': list(range(50))})
    table_name = 'my_test_table_s3'
    my_table = rh.table(data=data,
                        name=table_name,
                        data_url="donnyg-my-test-bucket/my_table.parquet",
                        data_source='s3'
                        )
    del data
    del my_table

    reloaded_table = rh.table(name=table_name, load_from=['rns'])
    assert reloaded_table.data['my_col'].to_pylist() == list(range(50))

    reloaded_table.delete_in_fs()
    assert not reloaded_table.folder.exists_in_fs()


if __name__ == '__main__':
    setup()
    unittest.main()
