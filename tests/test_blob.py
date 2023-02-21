import unittest
from pathlib import Path

import runhouse as rh
import yaml

from ray import cloudpickle as pickle

S3_BUCKET = "runhouse-tests"
TEMP_LOCAL_FOLDER = Path(__file__).parents[1] / "rh"


def test_create_and_reload_local_blob():
    name = "~/my_local_blob"
    data = pickle.dumps(list(range(50)))
    my_blob = rh.blob(
        data=data,
        name=name,
        path=str(TEMP_LOCAL_FOLDER / "my_blob.pickle"),
        system="file",
        dryrun=False,
    ).save()
    del data
    del my_blob

    reloaded_blob = rh.blob(name=name)
    reloaded_data = pickle.loads(reloaded_blob.data)
    assert reloaded_data == list(range(50))

    # Delete the blob itself
    reloaded_blob.delete_in_system()
    assert not reloaded_blob.exists_in_system()

    # Delete metadata saved locally and / or the database for the blob
    reloaded_blob.delete_configs()

    assert True


def test_create_and_reload_rns_blob():
    name = "@/my_s3_blob"
    data = pickle.dumps(list(range(50)))
    my_blob = rh.blob(
        name=name,
        data=data,
        system="s3",
        path=f"/{S3_BUCKET}/test_blob.pickle",
        mkdir=True,
        dryrun=False,
    ).save()

    del data
    del my_blob

    reloaded_blob = rh.blob(name=name)
    reloaded_data = pickle.loads(reloaded_blob.data)
    assert reloaded_data == list(range(50))

    # Delete the blob itself from the filesystem
    reloaded_blob.delete_in_system()
    assert not reloaded_blob.exists_in_system()

    # Delete metadata saved locally and / or the database for the blob and its associated folder
    reloaded_blob.delete_configs()


def test_from_cluster():
    cluster = rh.cluster(name="^rh-cpu").up_if_not()
    config_blob = rh.blob(path=rh.configs.CONFIG_PATH, system=cluster)
    config_data = yaml.safe_load(config_blob.data)
    assert len(config_data.keys()) > 4


def test_share():
    name = "@/s3_blob"
    data = pickle.dumps(list(range(50)))
    my_blob = rh.blob(
        data=data,
        name=name,
        path=f"/{S3_BUCKET}/test_blob.pickle",
        system="s3",
        dryrun=False,
    ).save()

    my_blob.share(
        users=["josh@run.house", "donny@run.house"], snapshot=False, access_type="write"
    )

    assert not my_blob.exists_in_system()


@unittest.skip("Needs to be run manually using a shared resource URI.")
def test_read_shared_blob():
    from runhouse import Blob

    my_blob = Blob.from_name("/<resource-sharer>/s3_blob")
    raw_data = my_blob.fetch()
    # NOTE: we need to do the deserialization ourselves
    assert pickle.loads(raw_data)


if __name__ == "__main__":
    unittest.main()
