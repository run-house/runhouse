import os
import unittest
from pathlib import Path

import pytest

import runhouse as rh
from ray import cloudpickle as pickle
from runhouse.globals import configs


DATA_STORE_BUCKET = "runhouse-folder"
DATA_STORE_PATH = f"/{DATA_STORE_BUCKET}/folder-tests"


def fs_str_rh_fn(folder):
    return folder._fs_str


# ----------------- Run tests -----------------


@pytest.mark.skip("Bad path")
@pytest.mark.clustertest
def test_from_cluster(cluster):
    rh.folder(path="../../../").to(cluster, path="my_new_tests_folder")
    tests_folder = rh.folder(system=cluster, path="my_new_tests_folder")
    assert "my_new_tests_folder/requirements.txt" in tests_folder.ls()


@pytest.mark.awstest
def test_create_and_save_data_to_s3_folder():
    data = list(range(50))
    s3_folder = rh.folder(path=DATA_STORE_PATH, system="s3")
    s3_folder.mkdir()
    s3_folder.put({"test_data.py": pickle.dumps(data)}, overwrite=True)

    assert s3_folder.exists_in_system()


@pytest.mark.awstest
def test_read_data_from_existing_s3_folder():
    # Note: Uses folder created above
    s3_folder = rh.folder(path=DATA_STORE_PATH, system="s3")
    fss_file: "fsspec.core.OpenFile" = s3_folder.open(name="test_data.py")
    with fss_file as f:
        data = pickle.load(f)

    assert data == list(range(50))


@pytest.mark.awstest
def test_create_and_delete_folder_from_s3():
    s3_folder = rh.folder(name=DATA_STORE_PATH, system="s3")
    s3_folder.mkdir()

    s3_folder.delete_configs()

    assert not s3_folder.exists_in_system()


@pytest.mark.clustertest
def test_folder_attr_on_cluster(local_folder, cluster):
    cluster_folder = local_folder.to(cluster)
    fs_str_cluster = rh.function(fn=fs_str_rh_fn).to(cluster)
    fs_str = fs_str_cluster(cluster_folder)
    assert fs_str == "file"


@pytest.mark.gcptest
@pytest.mark.awstest
@pytest.mark.clustertest
def test_cluster_tos(cluster, tmp_path):
    tests_folder = rh.folder(path=str(Path.cwd()))

    tests_folder = tests_folder.to(system=cluster)
    assert "test_folder.py" in tests_folder.ls(full_paths=False)

    # to local
    local = tests_folder.to("here", path=tmp_path)
    assert "test_folder.py" in local.ls(full_paths=False)

    # to s3
    s3 = tests_folder.to("s3")
    assert "test_folder.py" in s3.ls(full_paths=False)

    s3.rm()

    # to gcs
    gcs = tests_folder.to("gs")
    try:
        assert "test_folder.py" in gcs.ls(full_paths=False)
        gcs.rm()

    except:
        bucket_name = gcs._bucket_name_from_path(gcs.path)
        print(
            f"Permissions to gs bucket ({bucket_name}) may not be fully enabled "
            f"on the cluster {cluster.name}. For now please manually enable them directly on the cluster. "
            f"See https://cloud.google.com/sdk/gcloud/reference/auth/login"
        )


@pytest.mark.clustertest
def test_local_and_cluster(cluster, local_folder, tmp_path):
    # Local to cluster
    cluster_folder = local_folder.to(system=cluster)
    assert "sample_file_0.txt" in cluster_folder.ls(full_paths=False)
    assert cluster_folder._fs_str == "ssh"

    # Cluster to local
    local_from_cluster = cluster_folder.to("here", path=tmp_path)
    assert "sample_file_0.txt" in local_from_cluster.ls(full_paths=False)
    assert local_from_cluster._fs_str == "file"


@pytest.mark.awstest
def test_local_and_s3(local_folder, tmp_path):
    # Local to S3
    s3_folder = local_folder.to(system="s3")
    assert "sample_file_0.txt" in s3_folder.ls(full_paths=False)
    assert s3_folder._fs_str == "s3"

    # S3 to local
    local_from_s3 = s3_folder.to("here", path=tmp_path)
    assert "sample_file_0.txt" in local_from_s3.ls(full_paths=False)
    assert local_from_s3._fs_str == "file"

    s3_folder.rm()


@pytest.mark.gcptest
def test_local_and_gcs(local_folder, tmp_path):
    # Local to GCS
    gcs_folder = local_folder.to(system="gs")
    assert "sample_file_0.txt" in gcs_folder.ls(full_paths=False)
    assert gcs_folder._fs_str == "gs"

    # GCS to local
    local_from_gcs = gcs_folder.to("here", path=tmp_path)
    assert "sample_file_0.txt" in local_from_gcs.ls(full_paths=False)
    assert local_from_gcs._fs_str == "file"

    gcs_folder.rm()


@pytest.mark.awstest
@pytest.mark.clustertest
def test_cluster_and_s3(cluster, cluster_folder):
    # Cluster to S3
    s3_folder = cluster_folder.to(system="s3")
    assert "sample_file_0.txt" in s3_folder.ls(full_paths=False)
    assert s3_folder._fs_str == "s3"

    # S3 to cluster
    cluster_from_s3 = s3_folder.to(system=cluster)
    assert "sample_file_0.txt" in cluster_from_s3.ls(full_paths=False)
    assert cluster_from_s3._fs_str == "ssh"

    s3_folder.rm()


@unittest.skip("requires GCS setup")
@pytest.mark.clustertest
def test_cluster_and_gcs(cluster, cluster_folder):
    # Make sure we have gsutil and gcloud on the cluster - needed for copying the package + authenticating
    cluster.install_packages(["gsutil"])

    # TODO [JL] might be necessary to install gcloud on the cluster
    # c.run(['sudo snap install google-cloud-cli --classic'])

    # Cluster to GCS
    gcs_folder = cluster_folder.to(system="gs")
    try:
        assert "sample_file_0.txt" in gcs_folder.ls(full_paths=False)
        assert gcs_folder._fs_str == "gs"

        # GCS to cluster
        cluster_from_gcs = gcs_folder.to(system=cluster)
        assert "sample_file_0.txt" in cluster_from_gcs.ls(full_paths=False)
        assert cluster_from_gcs._fs_str == "ssh"

        gcs_folder.rm()

    except:
        # TODO [JL] automate gcloud access on the cluster for writing to GCS bucket
        bucket_name = gcs_folder._bucket_name_from_path(gcs_folder.path)
        raise PermissionError(
            f"Permissions to gs bucket ({bucket_name}) may not be fully enabled "
            f"on the cluster {cluster.name}. For now please manually enable them directly on the cluster. "
            f"See https://cloud.google.com/sdk/gcloud/reference/auth/login"
        )


@pytest.mark.awstest
def test_s3_and_s3(local_folder, s3_folder):
    # from one s3 folder to another s3 folder
    new_s3_folder = s3_folder.to(system="s3")
    assert "sample_file_0.txt" in new_s3_folder.ls(full_paths=False)

    new_s3_folder.rm()


@pytest.mark.gcptest
def test_gcs_and_gcs(gcs_folder):
    # from one gcs folder to another gcs folder
    new_gcs_folder = gcs_folder.to(system="gs")
    assert "sample_file_0.txt" in new_gcs_folder.ls(full_paths=False)

    new_gcs_folder.rm()


@pytest.mark.gcptest
@pytest.mark.awstest
@unittest.skip("Doesn't work properly as only full-bucket copy is supported")
def test_s3_and_gcs(s3_folder):
    # *** NOTE: transfers between providers are only supported at the bucket level at the moment (not directory) ***

    s3_folder_to_gcs = s3_folder.to(system="gs")
    assert "sample_file_0.txt" in s3_folder_to_gcs.ls(full_paths=False)
    assert s3_folder_to_gcs._fs_str == "gs"

    s3_folder_to_gcs.rm()


@pytest.mark.gcptest
@pytest.mark.awstest
@unittest.skip("Doesn't work properly as only full-bucket copy is supported")
def test_gcs_and_s3(local_folder, gcs_folder):
    # *** NOTE: transfers between providers are only supported at the bucket level at the moment (not directory) ***

    gcs_folder_to_s3 = gcs_folder.to(system="s3")
    assert "sample_file_0.txt" in gcs_folder_to_s3.ls(full_paths=False)
    assert gcs_folder_to_s3._fs_str == "s3"

    gcs_folder_to_s3.rm()


@pytest.mark.awstest
def test_s3_folder_uploads_and_downloads(local_folder, tmp_path):
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


@pytest.mark.clustertest
def test_cluster_and_cluster(byo_cpu, cluster, local_folder):
    # Upload sky secrets to cluster - required when syncing over the folder from c1 to c2
    byo_cpu.sync_secrets(providers=["sky"])

    cluster_folder_1 = local_folder.to(system=byo_cpu)
    assert "sample_file_0.txt" in cluster_folder_1.ls(full_paths=False)

    # Cluster 1 to cluster 2
    cluster_folder_2 = cluster_folder_1.to(system=cluster, path=cluster_folder_1.path)
    assert "sample_file_0.txt" in cluster_folder_2.ls(full_paths=False)

    # Cluster 2 to cluster 1
    cluster_folder_1.rm()
    cluster_folder_1 = cluster_folder_2.to(system=byo_cpu, path=cluster_folder_2.path)
    assert "sample_file_0.txt" in cluster_folder_1.ls(full_paths=False)


@pytest.mark.awstest
@pytest.mark.rnstest
def test_s3_sharing(s3_folder):
    token = os.getenv("TEST_TOKEN") or configs.get("token")
    headers = {"Authorization": f"Bearer {token}"}

    assert (
        token
    ), "No token provided. Either set `TEST_TOKEN` env variable or set `token` in the .rh config file"

    # Login to ensure the default folder / username are saved down correctly
    rh.login(token=token, download_config=True, interactive=False)

    s3_folder.save("@/my-s3-shared-folder")
    s3_folder.share(
        users=["donny@run.house", "josh@run.house"],
        access_level="read",
        notify_users=False,
        headers=headers,
    )

    my_folder = rh.folder(name="@/my-s3-shared-folder")
    assert my_folder.ls() == s3_folder.ls()


def test_github_folder(tmp_path):
    gh_folder = rh.folder(
        path="/", system="github", data_config={"org": "pytorch", "repo": "rfcs"}
    )
    assert gh_folder.ls()

    gh_to_local = gh_folder.to(
        system="file", path=tmp_path / "torchrfcs", data_config={}
    )
    assert (tmp_path / "torchrfcs").exists()
    assert "RFC-0000-template.md" in gh_to_local.ls(full_paths=False)


if __name__ == "__main__":
    unittest.main()
