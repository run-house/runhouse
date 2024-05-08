import pytest

import runhouse as rh

import tests.test_resources.test_resource

from ray import cloudpickle as pickle

DATA_STORE_BUCKET = "runhouse-folder"
DATA_STORE_PATH = f"/{DATA_STORE_BUCKET}/folder-tests"


def fs_str_rh_fn(folder):
    return folder._fs_str


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
    if folder_sys == "file" and dest_sys == "s3":
        pytest.skip("Built-in type region should not be set to None.")
    elif folder_sys == "gs" and dest_sys == "file":
        pytest.skip("Gsutil rsync command errors out.")


class TestFolder(tests.test_resources.test_resource.TestResource):

    MAP_FIXTURES = {"resource": "folder"}

    _unit_folder_fixtures = ["local_folder"]
    _local_folder_fixtures = _unit_folder_fixtures + ["local_folder_docker"]
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

    @pytest.mark.level("minimal")
    def test_send_folder_to_dest(self, folder, dest):
        _check_skip_test(folder, dest)

        system = "here" if dest.system == "file" else dest.system
        expected_fs_str = "ssh" if isinstance(dest.system, rh.Cluster) else dest.system

        new_folder = folder.to(system=system)

        assert new_folder._fs_str == expected_fs_str
        assert "sample_file_0.txt" in new_folder.ls(full_paths=False)

        new_folder.rm()

    @pytest.mark.level("local")
    @pytest.mark.skip("Bad path")
    def test_from_cluster(self, cluster):
        rh.folder(path="../../../").to(cluster, path="my_new_tests_folder")
        tests_folder = rh.folder(system=cluster, path="my_new_tests_folder")
        assert "my_new_tests_folder/requirements.txt" in tests_folder.ls()

    @pytest.mark.level("local")  # TODO: fix this test
    @pytest.mark.skip("[WIP] Fix this test")
    def test_folder_attr_on_cluster(self, local_folder, cluster):
        cluster_folder = local_folder.to(cluster)
        fs_str_cluster = rh.function(fn=fs_str_rh_fn).to(cluster)
        fs_str = fs_str_cluster(cluster_folder)
        assert fs_str == "file"

    @pytest.mark.level("unit")
    def test_github_folder(self, tmp_path):
        gh_folder = rh.folder(
            path="/", system="github", data_config={"org": "pytorch", "repo": "rfcs"}
        )
        assert gh_folder.ls()

        gh_to_local = gh_folder.to(
            system="file", path=tmp_path / "torchrfcs", data_config={}
        )
        assert (tmp_path / "torchrfcs").exists()
        assert "RFC-0000-template.md" in gh_to_local.ls(full_paths=False)

    ##### S3 Folder Tests #####
    @pytest.mark.level("minimal")
    def test_create_and_save_data_to_s3_folder(self):
        data = list(range(50))
        s3_folder = rh.folder(path=DATA_STORE_PATH, system="s3")
        s3_folder.mkdir()
        s3_folder.put({"test_data.py": pickle.dumps(data)}, overwrite=True)

        assert s3_folder.exists_in_system()

    @pytest.mark.level("minimal")
    def test_read_data_from_existing_s3_folder(self):
        # Note: Uses folder created above
        s3_folder = rh.folder(path=DATA_STORE_PATH, system="s3")
        fss_file: "fsspec.core.OpenFile" = s3_folder.open(name="test_data.py")
        with fss_file as f:
            data = pickle.load(f)

        assert data == list(range(50))

    @pytest.mark.level("minimal")
    def test_create_and_delete_folder_from_s3(self):
        s3_folder = rh.folder(name=DATA_STORE_PATH, system="s3")
        s3_folder.mkdir()

        s3_folder.delete_configs()
        s3_folder.rm()

        assert not s3_folder.exists_in_system()

    @pytest.mark.skip("Region needs to be supported for sending to s3.")
    @pytest.mark.level("minimal")  # TODO: needs S3 credentials
    def test_s3_folder_uploads_and_downloads(self, local_folder, tmp_path):
        # NOTE: you can also specify a specific path like this:
        # test_folder = rh.folder(path='/runhouse/my-folder', system='s3')

        s3_folder = rh.folder(system="s3")
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
