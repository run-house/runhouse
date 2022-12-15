import unittest
from pathlib import Path

import fsspec
from ray import cloudpickle as pickle

import runhouse as rh
import runhouse.rns.folders.folder
from runhouse.rh_config import rns_client
import runhouse.rns.top_level_rns_fns

TEMP_FILE = 'my_file.txt'
TEMP_FOLDER = 'runhouse-tests'


# TODO FAILS - where is tmp_dir being initialized?
def test_find_working_dir(tmp_path):
    tmp_path = '/Users/josh.l/dev/runhouse/rh/tmp'
    starting_dir = Path(tmp_path, 'subdir/subdir/subdir/subdir')
    d = rns_client.locate_working_dir(cwd=str(starting_dir))
    assert d in str(starting_dir)

    Path(tmp_path, 'subdir/rh').mkdir(parents=True)
    d = rns_client.locate_working_dir(str(starting_dir))
    assert d == str(Path(tmp_path, 'subdir'))

    Path(tmp_path, 'subdir/rh').rmdir()

    Path(tmp_path, 'subdir/subdir/.git').mkdir(exist_ok=True, parents=True)
    d = rns_client.locate_working_dir(str(starting_dir))
    assert d in str(Path(tmp_path, 'subdir/subdir'))

    Path(tmp_path, 'subdir/subdir/.git').rmdir()

    Path(tmp_path, 'subdir/subdir/requirements.txt').write_text('....')
    d = rns_client.locate_working_dir(str(starting_dir))
    assert d in str(Path(tmp_path, 'subdir/subdir'))


# TODO [JL / DG] FAILS
def test_set_folder(tmp_path):
    rh.folder('bert_ft', dryrun=False, save_to=['local'])
    runhouse.rns.top_level_rns_fns.set_folder('bert_ft')
    # assert rh.current_folder().url == str(Path.home() / 'runhouse/runhouse/rh/bert_ft')
    rh.folder(name='my_test_hw', dryrun=False, save_to=['local'])

    # TODO [DG] does this assume that the user must have runhouse in their home directory?
    assert (Path.home() / 'runhouse/runhouse/rh/bert_ft/my_test_hw').exists()
    assert runhouse.rns.top_level_rns_fns.exists('~/bert_ft/my_test_hw')


# TODO [JL / DG] FAILS
def test_contains(tmp_path):
    runhouse.rns.top_level_rns_fns.set_folder('~')
    assert rh.folder('bert_ft').contains('my_test_hw')

    assert rh.folder('bert_ft').contains('~/bert_ft/my_qtest_hw')

    runhouse.rns.top_level_rns_fns.set_folder('bert_ft')
    assert rh.folder('~/bert_ft').contains('./my_test_hw')


def test_rns_path(tmp_path):
    runhouse.rns.top_level_rns_fns.set_folder('~')

    assert rh.folder('bert_ft').rns_address == rh.configs.get('default_folder') + '/bert_ft'


# TODO [JL / DG] FAILS
def test_ls():
    rh.set_folder('~')
    assert rh.ls() == rh.ls('~/')
    assert rh.ls(full_paths=True)
    rh.set_folder('^')
    assert rh.ls() == ['rh-32-cpu', 'rh-gpu', 'rh-cpu', 'rh-4-gpu', 'rh-8-cpu',
                       'rh-v100', 'rh-8-v100', 'rh-8-gpu', 'rh-4-v100']
    assert rh.ls('bert_ft') == []  # We're still inside builtins so we can't see bert_ft
    assert rh.folder('~/bert_ft', dryrun=False).ls() == ['my_test_hw']
    rh.set_folder('~')
    assert rh.folder('bert_ft', dryrun=False).ls() == ['my_test_hw']
    assert rh.ls('bert_ft') == ['my_test_hw']


# TODO [JL / DG] FAILS
def test_github_folder():
    # TODO gh_folder = rh.folder(url='https://github.com/pytorch/pytorch', fs='github')
    gh_folder = rh.folder(url='/', fs='github', data_config={'org': 'pytorch',
                                                             'repo': 'pytorch'})
    assert gh_folder.ls()


def test_create_and_save_data_to_s3_folder():
    data = list(range(50))
    s3_folder = rh.folder(name=TEMP_FOLDER, fs='s3', dryrun=False)
    s3_folder.mkdir()
    s3_folder.put({TEMP_FILE: pickle.dumps(data)}, overwrite=True)

    assert s3_folder.exists_in_fs()


def test_read_data_from_existing_s3_folder():
    # Note: Uses folder created above
    s3_folder = rh.folder(name=TEMP_FOLDER, load_from=['rns'])
    fss_file: fsspec.core.OpenFile = s3_folder.get(name=TEMP_FILE)
    with fss_file as f:
        data = pickle.load(f)

    assert data == list(range(50))


def test_create_and_delete_folder_from_s3():
    s3_folder = rh.folder(name=TEMP_FOLDER, fs='s3', dryrun=False, save_to=['local', 'rns'])
    s3_folder.mkdir()

    # delete the folder from its relevant file system and its associated data saved locally and/or in the database
    s3_folder.delete_configs(delete_from=['local', 'rns'])
    s3_folder.delete_in_fs()

    assert not s3_folder.exists_in_fs()


if __name__ == '__main__':
    unittest.main()
