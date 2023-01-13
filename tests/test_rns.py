import unittest
from pathlib import Path

import runhouse as rh
import runhouse.rns.folders.folder
from runhouse.rh_config import rns_client
import runhouse.rns.top_level_rns_fns


def test_find_working_dir(tmp_path):
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


def test_set_folder(tmp_path):
    rh.folder('~/bert_ft', dryrun=False)
    rh.set_folder('bert_ft')
    rh.folder(name='~/my_test_hw', dryrun=False)

    # TODO [DG] does this assume that the user must have runhouse in their home directory?
    assert (Path(rh.rh_config.rns_client.rh_directory) / 'bert_ft/my_test_hw').exists()
    assert rh.exists('~/bert_ft/my_test_hw')


def test_contains(tmp_path):
    runhouse.rns.top_level_rns_fns.set_folder('~')
    assert rh.folder('bert_ft').contains('my_test_hw')

    assert rh.folder('bert_ft').contains('~/bert_ft/my_test_hw')

    runhouse.rns.top_level_rns_fns.set_folder('bert_ft')
    assert rh.folder('~/bert_ft').contains('./my_test_hw')


def test_rns_path(tmp_path):
    rh.set_folder('~')
    assert rh.folder('bert_ft').rns_address == '~/bert_ft'

    rh.set_folder('@')
    assert rh.folder('bert_ft').rns_address == rh.configs.get('default_folder') + '/bert_ft'


def test_ls():
    rh.set_folder('~')
    assert rh.resources() == rh.resources('~/')
    assert rh.resources(full_paths=True)
    rh.set_folder('^')
    assert rh.resources() == ['rh-32-cpu', 'rh-gpu', 'rh-cpu', 'rh-4-gpu', 'rh-8-cpu',
                              'rh-v100', 'rh-8-v100', 'rh-8-gpu', 'rh-4-v100']
    assert rh.resources('bert_ft') == []  # We're still inside builtins so we can't see bert_ft
    assert rh.folder('~/bert_ft', dryrun=False).resources() == ['my_test_hw']
    rh.set_folder('~')
    assert rh.folder('bert_ft', dryrun=False).resources() == ['my_test_hw']
    assert rh.resources('bert_ft') == ['my_test_hw']

def test_from_name():
    f = rh.Folder.from_name('~/bert_ft', dryrun=True)
    assert f.contains('my_test_hw')
    c = rh.Cluster.from_name('^rh-cpu', dryrun=True)
    assert c.instance_type == "m5.large"


if __name__ == '__main__':
    unittest.main()
