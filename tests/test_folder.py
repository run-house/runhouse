import os
import shutil
import unittest
from pathlib import Path

import pytest

import runhouse as rh
from ray import cloudpickle as pickle
from runhouse.rh_config import configs

TEST_FOLDER_PATH = Path.cwd() / "tests_tmp"

DATA_STORE_BUCKET = "runhouse-folder"
DATA_STORE_PATH = f"/{DATA_STORE_BUCKET}/folder-tests"


def setup():
    from pathlib import Path

    # Create buckets in S3 and GCS
    from runhouse.rns.api_utils.utils import create_s3_bucket

    create_s3_bucket(DATA_STORE_BUCKET)

    # TODO [JL] removing GCS tests for now
    # create_gcs_bucket(DATA_STORE_BUCKET)

    # Create local dir with files to upload to cluster, buckets, etc.
    TEST_FOLDER_PATH.mkdir(parents=True, exist_ok=True)
    for i in range(3):
        output_file = Path(f"{TEST_FOLDER_PATH}/sample_file_{i}.txt")
        output_file.write_text(f"file{i}")


def delete_local_folder(path):
    shutil.rmtree(path)


def fs_str_rh_fn(folder):
    return folder._fs_str


def test_github_folder():
    gh_folder = rh.folder(
        path="/", system="github", data_config={"org": "pytorch", "repo": "pytorch"}
    )
    assert gh_folder.ls()


# ----------------- Run tests -----------------


@pytest.mark.clustertest
def test_from_cluster(cpu_cluster):
    rh.folder(path=str(Path.cwd())).to(cpu_cluster, path="~/my_new_tests_folder")
    tests_folder = rh.folder(system=cpu_cluster, path="~/my_new_tests_folder")
    assert "my_new_tests_folder/test_folder.py" in tests_folder.ls()


@pytest.mark.awstest
@pytest.mark.clustertest
def test_to_cluster_attr(cpu_cluster, local_folder):
    cluster_folder = local_folder.to(system=cpu_cluster)
    assert isinstance(cluster_folder.system, rh.Cluster)
    assert cluster_folder._fs_str == "ssh"

    s3_folder = rh.folder(path=TEST_FOLDER_PATH, system="s3")
    cluster_folder_s3 = s3_folder.to(system=cpu_cluster)
    assert isinstance(cluster_folder_s3.system, rh.Cluster)
    assert cluster_folder_s3._fs_str == "ssh"


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
def test_folder_attr_on_cluster(cpu_cluster):
    cluster_folder = rh.folder(path=TEST_FOLDER_PATH).to(system=cpu_cluster)
    fs_str_cluster = rh.function(fn=fs_str_rh_fn).to(system=cpu_cluster)
    fs_str = fs_str_cluster(cluster_folder)
    assert fs_str == "file"


@pytest.mark.gcptest
@pytest.mark.awstest
@pytest.mark.clustertest
def test_cluster_tos(cpu_cluster):
    tests_folder = rh.folder(path=str(Path.cwd()))

    tests_folder = tests_folder.to(system=cpu_cluster)
    assert "test_folder.py" in tests_folder.ls(full_paths=False)

    # to local
    local = tests_folder.to("here", path=TEST_FOLDER_PATH)
    assert "test_folder.py" in local.ls(full_paths=False)

    # to s3
    s3 = tests_folder.to("s3")
    assert "test_folder.py" in s3.ls(full_paths=False)

    delete_local_folder(TEST_FOLDER_PATH)
    s3.delete_in_system()

    # to gcs
    gcs = tests_folder.to("gs")
    try:
        assert "test_folder.py" in gcs.ls(full_paths=False)
        gcs.delete_in_system()

    except:
        bucket_name = gcs.bucket_name_from_path(gcs.path)
        print(
            f"Permissions to gs bucket ({bucket_name}) may not be fully enabled "
            f"on the cluster {cpu_cluster.name}. For now please manually enable them directly on the cluster. "
            f"See https://cloud.google.com/sdk/gcloud/reference/auth/login"
        )


@pytest.mark.clustertest
def test_local_and_cluster(cpu_cluster, local_folder):
    # Local to cluster
    local_folder.mkdir()
    local_folder.put({f"sample_file_{i}.txt": f"file{i}".encode() for i in range(3)})
    cluster_folder = local_folder.to(system=cpu_cluster)
    assert "sample_file_0.txt" in cluster_folder.ls(full_paths=False)

    # Cluster to local
    tmp_path = Path.cwd() / "tmp_from_cluster"
    local_from_cluster = cluster_folder.to("here", path=tmp_path)
    assert "sample_file_0.txt" in local_from_cluster.ls(full_paths=False)

    delete_local_folder(tmp_path)


@pytest.mark.awstest
def test_local_and_s3(local_folder):
    # Local to S3
    s3_folder = local_folder.to(system="s3")
    assert "sample_file_0.txt" in s3_folder.ls(full_paths=False)

    # S3 to local
    tmp_path = Path.cwd() / "tmp_from_s3"
    tmp_path.mkdir(exist_ok=True)

    local_from_s3 = s3_folder.to("here", path=tmp_path)
    assert "sample_file_0.txt" in local_from_s3.ls(full_paths=False)

    delete_local_folder(tmp_path)
    s3_folder.delete_in_system()


@pytest.mark.gcptest
def test_local_and_gcs(local_folder):
    # Local to GCS
    gcs_folder = local_folder.to(system="gs")
    assert "sample_file_0.txt" in gcs_folder.ls(full_paths=False)

    # GCS to local
    tmp_path = Path.cwd() / "tmp_from_gcs"
    tmp_path.mkdir(exist_ok=True)

    local_from_gcs = gcs_folder.to("here", path=tmp_path)
    assert "sample_file_0.txt" in local_from_gcs.ls(full_paths=False)

    delete_local_folder(tmp_path)
    gcs_folder.delete_in_system()


@pytest.mark.awstest
@pytest.mark.clustertest
def test_cluster_and_s3(cpu_cluster, local_folder):
    # Local to cluster
    cluster_folder = local_folder.to(system=cpu_cluster)
    assert "sample_file_0.txt" in cluster_folder.ls(full_paths=False)

    # Cluster to S3
    s3_folder = cluster_folder.to(system="s3")
    assert "sample_file_0.txt" in s3_folder.ls(full_paths=False)

    # S3 to cluster
    cluster_from_s3 = s3_folder.to(system=cpu_cluster)
    assert "sample_file_0.txt" in cluster_from_s3.ls(full_paths=False)

    s3_folder.delete_in_system()


@unittest.skip("requires GCS setup")
@pytest.mark.clustertest
def test_cluster_and_gcs(cpu_cluster, local_folder):
    # Make sure we have gsutil and gcloud on the cluster - needed for copying the package + authenticating
    cpu_cluster.install_packages(["gsutil"])

    # TODO [JL] might be necessary to install gcloud on the cluster
    # c.run(['sudo snap install google-cloud-cli --classic'])

    cluster_folder = local_folder.to(system=cpu_cluster)
    assert "sample_file_0.txt" in cluster_folder.ls(full_paths=False)

    # Cluster to GCS
    gcs_folder = cluster_folder.to(system="gs")
    try:
        assert "sample_file_0.txt" in gcs_folder.ls(full_paths=False)
        # GCS to cluster
        cluster_from_gcs = gcs_folder.to(system=cpu_cluster)
        assert "sample_file_0.txt" in cluster_from_gcs.ls(full_paths=False)

        gcs_folder.delete_in_system()

    except:
        # TODO [JL] automate gcloud access on the cluster for writing to GCS bucket
        bucket_name = gcs_folder.bucket_name_from_path(gcs_folder.path)
        raise PermissionError(
            f"Permissions to gs bucket ({bucket_name}) may not be fully enabled "
            f"on the cluster {cpu_cluster.name}. For now please manually enable them directly on the cluster. "
            f"See https://cloud.google.com/sdk/gcloud/reference/auth/login"
        )


@pytest.mark.awstest
def test_s3_and_s3(local_folder):
    # Local to S3
    s3_folder = local_folder.to(system="s3")
    assert "sample_file_0.txt" in s3_folder.ls(full_paths=False)

    # from one s3 folder to another s3 folder
    new_s3_folder = s3_folder.to(system="s3")
    assert "sample_file_0.txt" in new_s3_folder.ls(full_paths=False)

    s3_folder.delete_in_system()
    new_s3_folder.delete_in_system()


@pytest.mark.gcptest
def test_gcs_and_gcs(local_folder):
    # Local to GCS
    gcs_folder = local_folder.to(system="gs")
    assert "sample_file_0.txt" in gcs_folder.ls(full_paths=False)

    # from one gcs folder to another gcs folder
    new_gcs_folder = gcs_folder.to(system="gs")
    assert "sample_file_0.txt" in new_gcs_folder.ls(full_paths=False)

    gcs_folder.delete_in_system()
    new_gcs_folder.delete_in_system()


@pytest.mark.gcptest
@pytest.mark.awstest
@unittest.skip("requires GCS setup")
def test_s3_and_gcs(local_folder):
    # Local to S3
    s3_folder = local_folder.to(system="s3")
    assert "sample_file_0.txt" in s3_folder.ls(full_paths=False)

    # *** NOTE: transfers between providers are only supported at the bucket level at the moment (not directory) ***

    # S3 to GCS
    s3_folder_to_gcs = s3_folder.to(system="gs")
    assert s3_folder_to_gcs.ls(full_paths=False)

    s3_folder.delete_in_system()


@pytest.mark.gcptest
@pytest.mark.awstest
@unittest.skip("requires GCS setup")
def test_gcs_and_s3(local_folder):
    # Local to GCS
    gcs_folder = local_folder.to(system="gs")
    assert "sample_file_0.txt" in gcs_folder.ls(full_paths=False)

    # *** NOTE: transfers between providers are only supported at the bucket level at the moment (not directory) ***

    # GCS to S3
    gcs_folder_to_s3 = gcs_folder.to(system="s3")
    assert gcs_folder_to_s3.ls(full_paths=False)

    gcs_folder.delete_in_system()


@pytest.mark.awstest
def test_s3_folder_uploads_and_downloads():
    # NOTE: you can also specify a specific path like this:
    # test_folder = rh.folder(path='/runhouse/my-folder', system='s3')

    s3_folder = rh.folder(system="s3")
    s3_folder.upload(src=str(TEST_FOLDER_PATH))

    assert s3_folder.exists_in_system()

    downloaded_path_folder = str(Path.cwd() / "downloaded_s3")
    s3_folder.download(dest=downloaded_path_folder)

    assert Path(downloaded_path_folder).exists()

    # remove folder in s3
    s3_folder.delete_in_system()
    assert not s3_folder.exists_in_system()


@pytest.mark.clustertest
def test_cluster_and_cluster(cpu_cluster, cpu_cluster_2, local_folder):
    # Upload sky secrets to cluster - required when syncing over the folder from c1 to c2
    cpu_cluster.send_secrets(providers=["sky"])

    cluster_folder_1 = local_folder.to(system=cpu_cluster)
    assert "sample_file_0.txt" in cluster_folder_1.ls(full_paths=False)

    # Cluster 1 to cluster 2
    c2 = rh.cluster(name=cpu_cluster_2).up_if_not()
    cluster_folder_2 = cluster_folder_1.to(system=c2, path=cluster_folder_1.path)
    assert "sample_file_0.txt" in cluster_folder_2.ls(full_paths=False)


@pytest.mark.awstest
@pytest.mark.rnstest
def test_s3_sharing():
    token = os.getenv("TEST_TOKEN") or configs.get("token")
    headers = {"Authorization": f"Bearer {token}"}

    assert (
        token
    ), "No token provided. Either set `TEST_TOKEN` env variable or set `token` in the .rh config file"

    # Login to ensure the default folder / username are saved down correctly
    rh.login(token=token, download_config=True, interactive=False)

    s3_folder = rh.folder(
        name="@/my-s3-shared-folder", path=DATA_STORE_PATH, system="s3"
    ).save()
    assert s3_folder.ls(full_paths=False)

    s3_folder.share(
        users=["donny@run.house", "josh@run.house"],
        access_type="read",
        notify_users=False,
        headers=headers,
    )

    assert s3_folder.ls(full_paths=False)


@pytest.mark.rnstest
def test_load_shared_folder():
    from runhouse import Folder

    my_folder = Folder.from_name("@/my-s3-shared-folder")
    folder_contents = my_folder.ls()
    assert folder_contents


if __name__ == "__main__":
    setup()
    unittest.main()
