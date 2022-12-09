import unittest
import pytest

import runhouse as rh

def test_create_and_reload(tmp_path):
    data = list(range(50))
    my_blob = rh.Blob(data=data,
                      name='my_test_blob',
                      data_url=str(tmp_path / "my_blob.pickle"),
                      data_source='file',
                      parent=tmp_path)
    del data
    del my_blob
    reloaded_blob = rh.Blob(name='my_test_blob', parent=tmp_path)
    assert reloaded_blob.data == list(range(50))

def test_create_and_reload_s3(tmp_path):
    data = list(range(50))
    blob_name = 'my_test_blob_s3'
    my_blob = rh.Blob(data=data,
                      name=blob_name,
                      data_url="donnyg-my-test-bucket/my_blob.pickle",
                      data_source='s3',
                      serializer='pickle',
                      parent=tmp_path
                      )
    del data
    del my_blob
    reloaded_blob = rh.Blob(name=blob_name, parent=tmp_path)
    assert reloaded_blob.data == list(range(50))
    reloaded_blob.delete_in_fs()
    assert not reloaded_blob.exists_in_fs()


if __name__ == '__main__':
    unittest.main()
