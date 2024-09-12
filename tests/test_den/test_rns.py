from pathlib import Path

import pytest

import runhouse as rh
from runhouse.utils import locate_working_dir


@pytest.mark.level("unit")
def test_find_working_dir(tmp_path):
    starting_dir = Path(tmp_path, "subdir/subdir/subdir/subdir")
    d = locate_working_dir(str(starting_dir))
    assert d in str(starting_dir)

    Path(tmp_path, "subdir/rh").mkdir(parents=True)
    d = locate_working_dir(str(starting_dir))
    assert d == str(Path(tmp_path, "subdir"))

    Path(tmp_path, "subdir/rh").rmdir()

    Path(tmp_path, "subdir/subdir/.git").mkdir(exist_ok=True, parents=True)
    d = locate_working_dir(str(starting_dir))
    assert d in str(Path(tmp_path, "subdir/subdir"))

    Path(tmp_path, "subdir/subdir/.git").rmdir()

    Path(tmp_path, "subdir/subdir/requirements.txt").write_text("....")
    d = locate_working_dir(str(starting_dir))
    assert d in str(Path(tmp_path, "subdir/subdir"))


def test_set_folder(tmp_path):
    rh.set_folder("~/tests")
    rh.folder(name="bert_ft").save()

    # TODO [DG] does this assume that the user must have runhouse in their home directory?
    assert rh.current_folder() == "~/tests"
    assert (Path(rh.globals.rns_client.rh_directory) / "tests/bert_ft").exists()
    assert rh.exists("~/tests/bert_ft")
    rh.set_folder("@")


def test_rns_path(tmp_path):
    rh.set_folder("~")
    assert rh.folder("tests").rns_address == "~/tests"

    rh.set_folder("@")
    assert (
        rh.folder("bert_ft").rns_address
        == rh.configs.get("default_folder") + "/bert_ft"
    )


@pytest.mark.skip(
    "TODO: [DG] This whole business is hanging on by a thread, we need to overhaul it."
)
def test_ls():
    rh.set_folder("~")
    assert rh.resources() == rh.resources("~")
    rh.set_folder("^")
    assert rh.resources() == [
        "rh-32-cpu",
        "rh-gpu",
        "rh-cpu",
        "rh-4-gpu",
        "rh-8-cpu",
        "rh-v100",
        "rh-8-v100",
        "rh-8-gpu",
        "rh-4-v100",
    ]
    assert (
        rh.resources("bert_ft") == []
    )  # We're still inside builtins so we can't see bert_ft
    assert "bert_ft" in rh.folder("~/tests").resources()
    rh.set_folder("~")
    assert "bert_ft" in rh.folder("tests").resources()
    assert "bert_ft" in rh.resources("tests")
    rh.set_folder("@")


def test_from_name(ondemand_aws_docker_cluster):
    f = rh.folder(name="~/tests/bert_ft")
    assert f.path
    assert ondemand_aws_docker_cluster.instance_type == "CPU:2+"
