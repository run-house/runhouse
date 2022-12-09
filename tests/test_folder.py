import unittest
from pathlib import Path

import runhouse as rh
import runhouse.rns.folder
from runhouse.rh_config import rns_client
import runhouse.rns.top_level_rns_fns


def test_find_working_dir(tmp_path):
    starting_dir = Path(tmp_path, 'subdir/subdir/subdir/subdir')
    starting_dir.mkdir(parents=True)
    d = rns_client.locate_working_dir(cwd=str(starting_dir))
    assert d is str(starting_dir)

    Path(tmp_path, 'subdir/rh').mkdir(parents=True)
    d = rns_client.locate_working_dir(str(starting_dir))
    assert d == str(Path(tmp_path, 'subdir'))

    Path(tmp_path, 'subdir/rh').rmdir()

    Path(tmp_path, 'subdir/subdir/.git').mkdir(exist_ok=True, parents=True)
    d = rns_client.locate_working_dir(str(starting_dir))
    assert d == str(Path(tmp_path, 'subdir/subdir'))

    Path(tmp_path, 'subdir/subdir/.git').rmdir()

    Path(tmp_path, 'subdir/subdir/subdir/requirements.txt').write_text('.')
    d = rns_client.locate_working_dir(str(starting_dir))
    assert d == str(Path(tmp_path, 'subdir/subdir/subdir'))

def test_set_folder(tmp_path):
    rh.folder('bert_ft')
    runhouse.rns.top_level_rns_fns.set_folder('bert_ft')
    # assert rh.current_folder().url == str(Path.home() / 'runhouse/runhouse/rh/bert_ft')
    rh.folder(name='my_test_hw')
    assert (Path.home() / 'runhouse/runhouse/rh/bert_ft/my_test_hw').exists()
    assert runhouse.rns.top_level_rns_fns.exists('~/bert_ft/my_test_hw')

def test_github_folder(tmp_path):
    # TODO gh_folder = rh.folder(url='https://github.com/pytorch/pytorch', fs='github')
    gh_folder = rh.folder(url='/', fs='github', data_config={'org': 'pytorch',
                                                             'repo': 'pytorch'})
    print(gh_folder.ls())

def test_s3_folder(tmp_path):
    # TODO
    # s3_folder = rh.folder(name='/my_folder', fs='s3')
    pass

def test_contains(tmp_path):
    runhouse.rns.top_level_rns_fns.set_folder('~')
    assert rh.folder('bert_ft').contains('my_test_hw')

    assert rh.folder('bert_ft').contains('~/bert_ft/my_test_hw')

    runhouse.rns.top_level_rns_fns.set_folder('bert_ft')
    assert rh.folder('~/bert_ft').contains('./my_test_hw')

def test_rns_path(tmp_path):
    runhouse.rns.top_level_rns_fns.set_folder('~')

    assert rh.folder('bert_ft').rns_address == rh.configs.get('default_folder') + '/bert_ft'

def test_ls(tmp_path):
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

if __name__ == '__main__':
    unittest.main()
