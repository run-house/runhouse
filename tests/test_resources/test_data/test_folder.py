import shutil
import tempfile
from pathlib import Path

import pytest

import runhouse as rh

import tests.test_resources.test_resource

from ray import cloudpickle as pickle

DATA_STORE_BUCKET = "runhouse-folder"
DATA_STORE_PATH = f"/{DATA_STORE_BUCKET}/folder-tests"


def _check_skip_test(folder, dest):
    folder_sys = (
        folder.system
        if not isinstance(folder.system, rh.Cluster)
        else folder.system.name
    )
    dest_sys = (
        dest.system if not isinstance(dest.system, rh.Cluster) else dest.system.name
    )
    systems_set = set()
    for system in folder_sys, dest_sys:
        system = (
            "docker" if "docker" in system else "rh-cpu" if "cpu" in system else system
        )
        systems_set.add(system)

    # Not supported
    if systems_set == {"s3", "gs"}:
        pytest.skip(
            "Transfer between S3 and GCS currently supported for buckets only, not specific directories"
        )

    # Improper cluster credentials setup
    if "rh-cpu" in systems_set:
        systems_set.remove("rh-cpu")
        if len(systems_set) > 0 and list(systems_set)[0] in [
            "docker",
            "s3",
            "gs",
        ]:
            pytest.skip(
                f"Cluster credentials for {list(systems_set)[0]} not set up properly."
            )
    elif "docker" in systems_set:
        systems_set.remove("docker")
        if len(systems_set) > 0 and list(systems_set)[0] in ["rh-cpu", "s3", "gs"]:
            pytest.skip(
                f"Docker cluster credentials for {list(systems_set)[0]} not set up properly."
            )

    # Bugs
    # Note: As of Jul-12-2024 doesn't seem to be an issue
    # if folder_sys == "file" and dest_sys == "s3":
    #     pytest.skip("Built-in type region should not be set to None.")
    if folder_sys == "gs" and dest_sys == "file":
        pytest.skip("Gsutil rsync command errors out.")


class TestFolder(tests.test_resources.test_resource.TestResource):

    MAP_FIXTURES = {"resource": "folder"}

    _unit_folder_fixtures = ["local_folder"]
    _local_folder_fixtures = _unit_folder_fixtures + ["docker_cluster_folder"]
    _all_folder_fixtures = _local_folder_fixtures + [
        "cluster_folder",
        "s3_folder",
        "gcs_folder",
    ]

    UNIT = {
        "folder": _unit_folder_fixtures,
        "dest": _unit_folder_fixtures,
    }
    LOCAL = {
        "folder": _local_folder_fixtures,
        "dest": _local_folder_fixtures,
    }
    MINIMAL = {
        "folder": _all_folder_fixtures,
        "dest": _all_folder_fixtures,
    }
    RELEASE = {
        "folder": _all_folder_fixtures,
        "dest": _all_folder_fixtures,
    }
    MAXIMAL = {
        "folder": _all_folder_fixtures,
        "dest": _all_folder_fixtures,
    }

    @pytest.mark.level("local")
    def test_cluster_folder_put_and_get_and_ls(self, docker_cluster_folder):
        docker_cluster_folder.put({"some_file.txt": "Hello world"})
        file_contents = docker_cluster_folder.get(name="some_file.txt")
        assert file_contents == "Hello world"

        folder_contents = docker_cluster_folder.ls(full_paths=False)
        assert "some_file.txt" in folder_contents

    @pytest.mark.level("local")
    def test_cluster_folder_exists_and_mkdir_and_rm(self, docker_cluster_folder):
        assert docker_cluster_folder.exists_in_system()
        docker_cluster_folder.rm()

        assert not docker_cluster_folder.exists_in_system()

        docker_cluster_folder.mkdir()
        assert docker_cluster_folder.exists_in_system()

    @pytest.mark.level("local")
    def test_cluster_folder_put_and_get_serialization_methods(
        self, docker_cluster_folder
    ):
        from runhouse.servers.http.http_utils import deserialize_data, serialize_data

        pickle_serialization = "pickle"
        raw_data = [1, 2, 3]
        docker_cluster_folder.put(
            {"some_file.pickle": serialize_data(raw_data, pickle_serialization)}
        )

        file_contents = docker_cluster_folder.get(name="some_file.pickle")
        assert deserialize_data(file_contents, pickle_serialization) == raw_data

        with pytest.raises(Exception):
            docker_cluster_folder.put({"some_file.pickle": pickle.dumps(raw_data)})

        json_serialization = "json"
        raw_data = {"name": "Runhouse"}
        docker_cluster_folder.put(
            {"some_file.text": serialize_data(raw_data, json_serialization)}
        )

        file_contents = docker_cluster_folder.get(name="some_file.text")
        assert deserialize_data(file_contents, json_serialization) == raw_data

    @pytest.mark.level("minimal")
    def test_send_folder_to_dest(self, folder, dest):
        _check_skip_test(folder, dest)

        system = "here" if dest.system == "file" else dest.system
        expected_fs_str = "ssh" if isinstance(dest.system, rh.Cluster) else dest.system

        new_folder = folder.to(system=system)

        assert new_folder._fs_str == expected_fs_str

        folder_contents = new_folder.ls(full_paths=False)
        assert "sample_file_0.txt" in folder_contents

        new_folder.rm()

    @pytest.mark.level("minimal")
    def test_cluster_folder_sync_upload(self, local_folder, cluster):
        cluster_dest = rh.Folder.DEFAULT_CACHE_FOLDER
        cluster.rsync(source=local_folder.path, dest=cluster_dest, up=True)
        res = cluster.run([f"ls -l {cluster_dest}"])

        assert all(f"sample_file_{i}.txt" in res[0][1] for i in range(3))

    @pytest.mark.level("minimal")
    def test_cluster_folder_sync_download(self, cluster_folder):
        import subprocess

        cluster = cluster_folder.system
        local_dest = rh.Folder.DEFAULT_CACHE_FOLDER

        cluster.rsync(
            source=cluster_folder.path, dest=local_dest, contents=True, up=False
        )
        res = subprocess.run(
            f"ls -l {local_dest}", shell=True, check=True, capture_output=True
        )

        assert all(f"sample_file_{i}.txt" in res.stdout.decode() for i in range(3))

    @pytest.mark.level("minimal")
    def test_send_folder_to_cluster(self, cluster):
        path = Path.cwd()
        local_folder = rh.folder(path=path)

        # Send the folder to the cluster, receive a new folder object in return which points to cluster's file system
        cluster_folder = local_folder.to(cluster)
        assert cluster_folder.system == cluster

        # Add a new file to the folder on the cluster
        cluster_folder.put({"requirements.txt": "torch"})
        folder_contents = cluster_folder.ls()
        res = [f for f in folder_contents if "requirements.txt" in f]
        assert res

    ##### S3 Folder Tests #####
    @pytest.mark.level("minimal")
    def test_send_local_folder_to_s3(self):
        data = list(range(50))

        # set initially to local file system, then send to s3
        path = Path.cwd()
        local_folder = rh.folder(path=path)
        assert local_folder.system == "file"

        s3_folder = local_folder.to("s3")
        assert s3_folder.system == "s3"

        s3_folder.put({"test_data.py": pickle.dumps(data)}, overwrite=True)
        assert s3_folder.exists_in_system()

        s3_folder.rm()
        assert not s3_folder.exists_in_system()

    @pytest.mark.level("minimal")
    def test_save_local_folder_to_s3(self):
        temp_dir = tempfile.mkdtemp()
        try:
            data = list(range(50))
            fake_file_path = Path(temp_dir) / "test_data.py"
            with open(fake_file_path, "wb") as f:
                pickle.dump(data, f)

            local_folder = rh.folder(path=fake_file_path.parent)
            assert local_folder.system == "file"

            s3_folder = local_folder.to("s3", path=DATA_STORE_PATH)
            assert s3_folder.system == "s3"
            assert s3_folder.exists_in_system()

        finally:
            shutil.rmtree(temp_dir)

    @pytest.mark.level("minimal")
    def test_read_data_from_existing_s3_folder(self):
        # Note: here we initialize the folder with the s3 system
        s3_folder = rh.folder(path=DATA_STORE_PATH, system="s3")

        file_name = "test_data.py"
        file_stream = s3_folder.open(name=file_name)
        with file_stream as f:
            data = pickle.load(f)

        assert data == list(range(50))

        file_contents = s3_folder.get(file_name)
        assert isinstance(file_contents, bytes)

    @pytest.mark.level("minimal")
    def test_create_and_delete_folder_from_s3(self):
        s3_folder = rh.folder(name=DATA_STORE_PATH, system="s3")
        s3_folder.mkdir()

        s3_folder.delete_configs()
        s3_folder.rm()

        assert not s3_folder.exists_in_system()

    @pytest.mark.level("minimal")
    def test_s3_folder_uploads_and_downloads(self, local_folder, tmp_path):
        s3_folder = rh.folder(system="s3")
        assert s3_folder.system == "s3"

        s3_folder._upload(src=local_folder.path)

        assert s3_folder.exists_in_system()
        assert "sample_file_0.txt" in s3_folder.ls(full_paths=False)

        downloaded_path_folder = tmp_path / "downloaded_s3"
        s3_folder._download(dest=downloaded_path_folder)

        assert downloaded_path_folder.exists()
        assert "sample_file_0.txt" in rh.folder(path=downloaded_path_folder).ls(
            full_paths=False
        )

        # remove folder in s3
        s3_folder.rm()
        assert not s3_folder.exists_in_system()
