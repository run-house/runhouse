import unittest
from pathlib import Path

from ray import cloudpickle as pickle
import runhouse as rh

S3_BUCKET = 'runhouse-tests'
TEMP_LOCAL_FOLDER = Path(__file__).parents[1] / 'rh'


def test_create_and_reload_local_blob():
    name = 'my_local_blob'
    data = pickle.dumps(list(range(50)))
    my_blob = rh.blob(data=data,
                      name=name,
                      url=str(TEMP_LOCAL_FOLDER / "my_blob.pickle"),
                      data_source='file',
                      save_to=['local'],
                      dryrun=False)
    del data
    del my_blob

    reloaded_blob = rh.blob(name=name, load_from=['local'])
    assert reloaded_blob.data == list(range(50))

    # Delete the blob itself
    reloaded_blob.delete_in_fs()
    assert not reloaded_blob.exists_in_fs()

    # Delete metadata saved locally and / or the database for the blob
    reloaded_blob.delete_configs(delete_from=['local'])

    assert True


def test_create_and_reload_rns_blob():
    # Make sure folder where the blob will be saved exists in the remote filesystem (e.g. S3)
    rh.folder(name=S3_BUCKET, fs='s3').mkdir()

    name = "my_s3_blob"
    data = pickle.dumps(list(range(50)))
    url = f'/{S3_BUCKET}/test_blob.pickle'
    my_blob = rh.blob(name=name,
                      data=data,
                      url=url,
                      data_source='s3',
                      save_to=['rns'],
                      dryrun=False
                      )

    reloaded_blob = rh.blob(name=name, load_from=['rns'])
    assert reloaded_blob.data == list(range(50))

    # Delete the blob itself from the filesystem
    reloaded_blob.delete_in_fs()
    assert not reloaded_blob.exists_in_fs()

    # TODO [JL] Also delete the folder which contained the blob?
    # reloaded_blob.folder.delete_in_fs()
    # assert not reloaded_blob.folder.exists_in_fs()

    # Delete metadata saved locally and / or the database for the blob and its associated folder
    reloaded_blob.delete_configs(delete_from=['rns'])


if __name__ == '__main__':
    unittest.main()
