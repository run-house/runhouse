from pathlib import Path

import pytest

import runhouse as rh

from runhouse import Cluster


TEMP_LOCAL_FOLDER = Path(__file__).parents[1] / "rh-blobs"


def test_save_local_blob_fails(local_blob, blob_data):
    with pytest.raises(ValueError):
        local_blob.save(name="my_local_blob")


@pytest.mark.parametrize(
    "blob",
    ["local_file", "s3_blob", "gcs_blob"],
    indirect=True,
)
def test_reload_blob_with_name(blob):
    name = "my_blob"
    blob.save(name)
    original_system = str(blob.system)
    original_data_str = str(blob.fetch())

    del blob

    reloaded_blob = rh.blob(name=name)
    assert str(reloaded_blob.system) == str(original_system)
    reloaded_data = reloaded_blob.fetch()
    assert reloaded_data[1] == "test"
    assert str(reloaded_data) == original_data_str

    # Delete metadata saved locally and / or the database for the blob
    reloaded_blob.delete_configs()

    # Delete the blob
    reloaded_blob.rm()
    assert not reloaded_blob.exists_in_system()


@pytest.mark.parametrize(
    "blob", ["local_file", "s3_blob", "gcs_blob", "cluster_file"], indirect=True
)
def test_reload_file_with_path(blob):
    reloaded_blob = rh.blob(path=blob.path, system=blob.system)
    reloaded_data = reloaded_blob.fetch()
    assert reloaded_data[1] == "test"

    # Delete the blob
    reloaded_blob.rm()
    assert not reloaded_blob.exists_in_system()


@pytest.mark.parametrize("file", ["local_file", "cluster_file"], indirect=True)
def test_file_to_blob(file, cluster):
    local_blob = file.to("here")
    assert local_blob.system is None
    fetched = local_blob.fetch()
    assert fetched[1] == "test"
    assert str(fetched) == str(file.fetch())

    cluster_blob = file.to(cluster)
    assert isinstance(cluster_blob.system, Cluster)
    fetched = cluster_blob.fetch()
    assert fetched[1] == "test"
    assert str(fetched) == str(file.fetch())


@pytest.mark.parametrize(
    "blob", ["local_blob", "cluster_blob", "local_file"], indirect=True
)
@pytest.mark.parametrize(
    "folder",
    ["local_folder", "cluster_folder", "s3_folder", "gcs_folder"],
    indirect=True,
)
def test_blob_to_file(blob, folder):
    new_file = blob.to(
        system=folder.system,
        path=folder.path + "/test_blob.pickle",
        data_config=folder.data_config,
    )
    assert new_file.system == folder.system
    assert new_file.path == folder.path + "/test_blob.pickle"
    fetched = new_file.fetch()
    assert fetched[1] == "test"
    assert str(fetched) == str(blob.fetch())
    assert "test_blob.pickle" in folder.ls(full_paths=False)


@pytest.mark.skip
def test_sharing_blob(cluster_blob):
    pass
    # TODO


def test_load_shared_blob(local_blob):
    my_blob = rh.blob(name="@/shared_blob")
    assert my_blob.exists_in_system()

    reloaded_data = my_blob.fetch()
    # NOTE: we need to do the deserialization ourselves
    assert str(reloaded_data) == str(local_blob.fetch())
