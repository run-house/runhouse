import unittest
from pathlib import Path

import fsspec
from ray import cloudpickle as pickle

import runhouse as rh

TEMP_FILE = 'my_file.txt'
TEMP_FOLDER = '~/runhouse-tests'


def test_github_folder():
    # TODO gh_folder = rh.folder(url='https://github.com/pytorch/pytorch', fs='github')
    gh_folder = rh.folder(url='/', fs='github', data_config={'org': 'pytorch',
                                                             'repo': 'pytorch'})
    assert gh_folder.ls()


def test_from_cluster():
    # Assumes a rh-cpu is already up from another test
    cluster = rh.cluster(name='^rh-cpu').up_if_not()
    rh.folder(url=str(Path.cwd())).to(cluster, url='~/my_new_tests_folder')
    tests_folder = rh.folder(fs='file', url='~/my_new_tests_folder').from_cluster(cluster)
    assert 'my_new_tests_folder/test_folder.py' in tests_folder.ls()


def test_create_and_save_data_to_s3_folder():
    data = list(range(50))
    s3_folder = rh.folder(name=TEMP_FOLDER, fs='s3', dryrun=False)
    s3_folder.mkdir()
    s3_folder.put({TEMP_FILE: pickle.dumps(data)}, overwrite=True)

    assert s3_folder.exists_in_fs()


def test_read_data_from_existing_s3_folder():
    # Note: Uses folder created above
    s3_folder = rh.folder(name=TEMP_FOLDER)
    fss_file: fsspec.core.OpenFile = s3_folder.open(name=TEMP_FILE)
    with fss_file as f:
        data = pickle.load(f)

    assert data == list(range(50))


def test_create_and_delete_folder_from_s3():
    s3_folder = rh.folder(name=TEMP_FOLDER, fs='s3', dryrun=False)
    s3_folder.mkdir()

    # delete the folder from its relevant file system and its associated data saved locally and/or in the database
    s3_folder.delete_configs()
    s3_folder.delete_in_fs()

    assert not s3_folder.exists_in_fs()

def test_cluster_tos(tmp_path):
    tests_folder = rh.folder(url=str(Path.cwd()))

    c = rh.cluster('^rh-cpu').up_if_not()
    # TODO [DG] change default behavior to return the from_cluster folder
    tests_folder = tests_folder.to(fs=c).from_cluster(c)
    assert 'test_folder.py' in tests_folder.ls(full_paths=False)

    # to local
    local = tests_folder.to('here', url=tmp_path)
    assert 'test_folder.py' in local.ls(full_paths=False)

    # to s3
    s3 = tests_folder.to('s3')
    assert 'test_folder.py' in s3.ls(full_paths=False)

    # to gcs
    gcs = tests_folder.to('gcs')
    assert 'test_folder.py' in gcs.ls(full_paths=False)

    # TODO to azure or R2


if __name__ == '__main__':
    unittest.main()
