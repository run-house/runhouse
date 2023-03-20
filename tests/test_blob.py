import unittest
from pathlib import Path

import runhouse as rh
import yaml

from ray import cloudpickle as pickle

S3_BUCKET = "runhouse-blob"
TEMP_LOCAL_FOLDER = Path(__file__).parents[1] / "rh-blobs"


def setup():
    from runhouse.rns.api_utils.utils import create_s3_bucket

    create_s3_bucket(S3_BUCKET)


def test_create_and_reload_local_blob_with_name():
    name = "~/my_local_blob"
    data = pickle.dumps(list(range(50)))
    my_blob = (
        rh.blob(
            data=data,
            name=name,
            system="file",
        )
        .write()
        .save()
    )

    del data
    del my_blob

    reloaded_blob = rh.Blob.from_name(name)
    reloaded_data = pickle.loads(reloaded_blob.data)
    assert reloaded_data == list(range(50))

    # Delete metadata saved locally and / or the database for the blob
    reloaded_blob.delete_configs()

    # Delete the blob
    reloaded_blob.delete_in_system()
    assert not reloaded_blob.exists_in_system()


def test_create_and_reload_local_blob_with_path():
    name = "~/my_local_blob"
    data = pickle.dumps(list(range(50)))
    my_blob = (
        rh.blob(
            data=data,
            name=name,
            path=str(TEMP_LOCAL_FOLDER / "my_blob.pickle"),
            system="file",
        )
        .write()
        .save()
    )

    del data
    del my_blob

    reloaded_blob = rh.Blob.from_name(name)
    reloaded_data = pickle.loads(reloaded_blob.data)
    assert reloaded_data == list(range(50))

    # Delete metadata saved locally and / or the database for the blob
    reloaded_blob.delete_configs()

    # Delete just the blob itself - since we define a custom path to store the blob, we want to keep the other
    # files stored in that directory
    reloaded_blob.delete_in_system()
    assert not reloaded_blob.exists_in_system()


def test_create_and_reload_anom_local_blob():
    data = pickle.dumps(list(range(50)))
    my_blob = rh.blob(
        data=data,
        system="file",
    ).write()

    reloaded_blob = rh.blob(path=my_blob.path, dryrun=True)
    reloaded_data = pickle.loads(reloaded_blob.data)
    assert reloaded_data == list(range(50))

    # Delete the blob
    reloaded_blob.delete_in_system()
    assert not reloaded_blob.exists_in_system()


def test_create_and_reload_rns_blob():
    name = "@/s3_blob"
    data = pickle.dumps(list(range(50)))
    my_blob = (
        rh.blob(
            name=name,
            data=data,
            system="s3",
            mkdir=True,
        )
        .write()
        .save()
    )

    del data
    del my_blob

    reloaded_blob = rh.Blob.from_name(name)
    reloaded_data = pickle.loads(reloaded_blob.data)
    assert reloaded_data == list(range(50))

    # Delete metadata saved locally and / or the database for the blob and its associated folder
    reloaded_blob.delete_configs()

    # Delete the blob itself from the filesystem
    reloaded_blob.delete_in_system()
    assert not reloaded_blob.exists_in_system()


def test_create_and_reload_rns_blob_with_path():
    name = "@/s3_blob"
    data = pickle.dumps(list(range(50)))
    my_blob = (
        rh.blob(
            name=name,
            data=data,
            system="s3",
            path=f"/{S3_BUCKET}/test_blob.pickle",
            mkdir=True,
        )
        .write()
        .save()
    )

    del data
    del my_blob

    reloaded_blob = rh.Blob.from_name(name)
    reloaded_data = pickle.loads(reloaded_blob.data)
    assert reloaded_data == list(range(50))

    # Delete metadata saved locally and / or the database for the blob and its associated folder
    reloaded_blob.delete_configs()

    # Delete the blob itself from the filesystem
    reloaded_blob.delete_in_system()
    assert not reloaded_blob.exists_in_system()


def test_to_cluster_attr():
    cluster = rh.cluster(name="^rh-cpu").up_if_not()
    local_blob = rh.blob(pickle.dumps(list(range(50))), path="models/pipeline.pkl")
    cluster_blob = local_blob.to(system=cluster)
    assert isinstance(cluster_blob.system, rh.Cluster)
    assert cluster_blob._folder._fs_str == "ssh"


def test_local_to_cluster():
    name = "~/my_local_blob"
    data = pickle.dumps(list(range(50)))
    my_blob = (
        rh.blob(
            data=data,
            name=name,
            system="file",
        )
        .write()
        .save()
    )

    cluster = rh.cluster(name="^rh-cpu").up_if_not()
    my_blob = my_blob.to(system=cluster)
    blob_data = pickle.loads(my_blob.data)
    assert blob_data == list(range(50))


def test_save_blob_to_cluster():
    cluster = rh.cluster(name="^rh-cpu").up_if_not()
    # Save blob to local directory, then upload to a new "models" directory on the root path of the cluster
    rh.blob(pickle.dumps(list(range(50))), path="models/pipeline.pkl").to(
        cluster, path="models"
    )

    # Confirm the model is saved on the cluster in the `models` folder
    status_codes = cluster.run(commands=["ls models"])
    assert "pipeline.pkl" in status_codes[0][1]


def test_from_cluster():
    cluster = rh.cluster(name="^rh-cpu").up_if_not()
    config_blob = rh.blob(path="/home/ubuntu/.rh/config.yaml", system=cluster)
    config_data = yaml.safe_load(config_blob.data)
    assert len(config_data.keys()) > 4


def test_sharing_blob():
    data = pickle.dumps(list(range(50)))
    name = "shared_blob"

    my_blob = (
        rh.blob(
            data=data,
            name=name,
            system="s3",
            mkdir=True,
        )
        .write()
        .save()
    )

    my_blob.share(
        users=["donny@run.house", "josh@run.house"],
        access_type="write",
        notify_users=False,
    )

    assert my_blob.exists_in_system()


def test_load_shared_blob():
    my_blob = rh.Blob.from_name(name="@/shared_blob")
    assert my_blob.exists_in_system()

    raw_data = my_blob.fetch()
    # NOTE: we need to do the deserialization ourselves
    assert pickle.loads(raw_data)


def test_save_anom_blob_to_s3():
    data = pickle.dumps(list(range(50)))
    my_blob = rh.blob(
        data=data,
        system="s3",
    ).write()

    assert my_blob.exists_in_system()


if __name__ == "__main__":
    setup()
    unittest.main()
