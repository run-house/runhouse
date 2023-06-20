import os
import unittest
from pathlib import Path

import pytest

import runhouse as rh
import yaml

from ray import cloudpickle as pickle

from runhouse.rh_config import configs

TEMP_LOCAL_FOLDER = Path(__file__).parents[1] / "rh-blobs"


@pytest.mark.rnstest
def test_create_and_reload_local_blob_with_name(blob_data):
    name = "~/my_local_blob"
    my_blob = (
        rh.blob(
            data=blob_data,
            name=name,
            system="file",
        )
        .write()
        .save()
    )

    del blob_data
    del my_blob

    reloaded_blob = rh.blob(name=name)
    reloaded_data = pickle.loads(reloaded_blob.data)
    assert reloaded_data == list(range(50))

    # Delete metadata saved locally and / or the database for the blob
    reloaded_blob.delete_configs()

    # Delete the blob
    reloaded_blob.rm()
    assert not reloaded_blob.exists_in_system()


@pytest.mark.rnstest
def test_create_and_reload_local_blob_with_path(blob_data):
    name = "~/my_local_blob"
    my_blob = (
        rh.blob(
            data=blob_data,
            name=name,
            path=str(TEMP_LOCAL_FOLDER / "my_blob.pickle"),
            system="file",
        )
        .write()
        .save()
    )

    del blob_data
    del my_blob

    reloaded_blob = rh.blob(name=name)
    reloaded_data = pickle.loads(reloaded_blob.data)
    assert reloaded_data == list(range(50))

    # Delete metadata saved locally and / or the database for the blob
    reloaded_blob.delete_configs()

    # Delete just the blob itself - since we define a custom path to store the blob, we want to keep the other
    # files stored in that directory
    reloaded_blob.rm()
    assert not reloaded_blob.exists_in_system()


@pytest.mark.localtest
def test_create_and_reload_anom_local_blob(blob_data):
    my_blob = rh.blob(
        data=blob_data,
        system="file",
    ).write()

    reloaded_blob = rh.blob(path=my_blob.path)
    reloaded_data = pickle.loads(reloaded_blob.data)
    assert reloaded_data == list(range(50))

    # Delete the blob
    reloaded_blob.rm()
    assert not reloaded_blob.exists_in_system()


@pytest.mark.awstest
@pytest.mark.rnstest
def test_create_and_reload_rns_blob(blob_data):
    name = "@/s3_blob"
    my_blob = (
        rh.blob(
            name=name,
            data=blob_data,
            system="s3",
        )
        .write()
        .save()
    )

    del blob_data
    del my_blob

    reloaded_blob = rh.blob(name=name)
    reloaded_data = pickle.loads(reloaded_blob.data)
    assert reloaded_data == list(range(50))

    # Delete metadata saved locally and / or the database for the blob and its associated folder
    reloaded_blob.delete_configs()

    # Delete the blob itself from the filesystem
    reloaded_blob.rm()
    assert not reloaded_blob.exists_in_system()


@pytest.mark.awstest
@pytest.mark.rnstest
def test_create_and_reload_rns_blob_with_path(blob_data, blob_s3_bucket):
    name = "@/s3_blob"
    my_blob = (
        rh.blob(
            name=name,
            data=blob_data,
            system="s3",
            path=f"/{blob_s3_bucket}/test_blob.pickle",
        )
        .write()
        .save()
    )

    del blob_data
    del my_blob

    reloaded_blob = rh.blob(name=name)
    reloaded_data = pickle.loads(reloaded_blob.data)
    assert reloaded_data == list(range(50))

    # Delete metadata saved locally and / or the database for the blob and its associated folder
    reloaded_blob.delete_configs()

    # Delete the blob itself from the filesystem
    reloaded_blob.rm()
    assert not reloaded_blob.exists_in_system()


@pytest.mark.clustertest
def test_to_cluster_attr(cpu_cluster, tmp_path):
    local_blob = rh.blob(
        pickle.dumps(list(range(50))), path=str(tmp_path / "pipeline.pkl")
    )
    cluster_blob = local_blob.to(system=cpu_cluster)
    assert isinstance(cluster_blob.system, rh.Cluster)
    assert cluster_blob._folder._fs_str == "ssh"


@pytest.mark.clustertest
@pytest.mark.rnstest
def test_local_to_cluster(cpu_cluster, blob_data):
    name = "~/my_local_blob"
    my_blob = (
        rh.blob(
            data=blob_data,
            name=name,
            system="file",
        )
        .write()
        .save()
    )

    my_blob = my_blob.to(system=cpu_cluster)
    blob_data = pickle.loads(my_blob.data)
    assert blob_data == list(range(50))


@pytest.mark.clustertest
def test_save_blob_to_cluster(cpu_cluster, tmp_path):
    # Save blob to local directory, then upload to a new "models" directory on the root path of the cluster
    model = rh.blob(
        pickle.dumps(list(range(50))), path=str(tmp_path / "pipeline.pkl")
    ).write()
    model.to(cpu_cluster, path="models")

    # Confirm the model is saved on the cluster in the `models` folder
    status_codes = cpu_cluster.run(commands=["ls models"])
    assert "pipeline.pkl" in status_codes[0][1]


@pytest.mark.clustertest
def test_from_cluster(cpu_cluster):
    config_blob = rh.blob(path="/home/ubuntu/.rh/config.yaml", system=cpu_cluster)
    config_data = yaml.safe_load(config_blob.data)
    assert len(config_data.keys()) > 4


@pytest.mark.awstest
@pytest.mark.rnstest
def test_sharing_blob(blob_data):
    token = os.getenv("TEST_TOKEN") or configs.get("token")
    headers = {"Authorization": f"Bearer {token}"}

    assert (
        token
    ), "No token provided. Either set `TEST_TOKEN` env variable or set `token` in the .rh config file"

    # Login to ensure the default folder / username are saved down correctly
    rh.login(token=token, download_config=True, interactive=False)

    name = "@/shared_blob"

    my_blob = (
        rh.blob(
            data=blob_data,
            name=name,
            system="s3",
        )
        .write()
        .save()
    )

    my_blob.share(
        users=["donny@run.house", "josh@run.house"],
        access_type="write",
        notify_users=False,
        headers=headers,
    )

    assert my_blob.exists_in_system()


@pytest.mark.rnstest
def test_load_shared_blob():
    my_blob = rh.blob(name="@/shared_blob")
    assert my_blob.exists_in_system()

    raw_data = my_blob.fetch()
    # NOTE: we need to do the deserialization ourselves
    assert pickle.loads(raw_data)


@pytest.mark.awstest
@pytest.mark.rnstest
def test_save_anom_blob_to_s3(blob_data):
    my_blob = rh.blob(
        data=blob_data,
        system="s3",
    ).write()

    assert my_blob.exists_in_system()


if __name__ == "__main__":
    unittest.main()
