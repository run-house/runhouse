import unittest
from pathlib import Path

import runhouse as rh

TEMP_LOCAL_FOLDER = Path(__file__).parents[1] / 'rh'
FOLDER = '/myblob/runhouse-blob'


def test_create_and_reload():
    data = list(range(50))
    my_blob = rh.blob(data=data,
                      name='my_test_blob',
                      data_url=str(TEMP_LOCAL_FOLDER / "my_blob.pickle"),
                      data_source='file',
                      dryrun=False)
    del data
    del my_blob

    reloaded_blob = rh.blob(name='my_test_blob', load_from=['local'])
    assert reloaded_blob.data == list(range(50))

    # Delete the blob itself
    reloaded_blob.delete_in_fs()
    assert not reloaded_blob.exists_in_fs()

    # Delete metadata saved locally and / or the database for the blob
    reloaded_blob.delete_configs(delete_from=['local'])

    assert True


def test_create_and_reload_blob_from_s3():
    data = list(range(50))
    blob_name = 'my_test_blob_s3'
    my_blob = rh.blob(folder=FOLDER,
                      data=data,
                      name=blob_name,
                      data_url=f'{FOLDER}/my_blob.pickle',
                      data_source='s3',
                      save_to=['rns'],
                      dryrun=False
                      )
    del data
    del my_blob

    reloaded_blob = rh.blob(name=blob_name, load_from=['rns'])
    assert reloaded_blob.data == list(range(50))

    # Delete the blob itself from the filesystem
    reloaded_blob.delete_in_fs()
    assert not reloaded_blob.exists_in_fs()

    # Delete the folder which contained the blob
    reloaded_blob.folder.delete_in_fs()
    assert not reloaded_blob.folder.exists_in_fs()

    # Delete metadata saved locally and / or the database for the blob and its associated folder
    reloaded_blob.delete_configs(delete_from=['local', 'rns'])

    assert True


if __name__ == '__main__':
    unittest.main()
