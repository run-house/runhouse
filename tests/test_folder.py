import shutil
import unittest
from pathlib import Path

import runhouse as rh
from ray import cloudpickle as pickle

TEMP_FILE = str(Path.cwd() / "test_folder.py")
TEST_FOLDER_PATH = Path.cwd() / "tests_tmp"

DATA_STORE_BUCKET = "/runhouse-folder-tests"
DATA_STORE_PATH = f"{DATA_STORE_BUCKET}/folder"


def setup():
    # Create buckets in S3 and GCS
    from sky.data.storage import GcsStore, S3Store

    S3Store(name="runhouse", source="")
    GcsStore(name="runhouse", source="")

    # Create local dir with files to upload to cluster, buckets, etc.
    from pathlib import Path

    TEST_FOLDER_PATH.mkdir(exist_ok=True)
    for i in range(3):
        output_file = Path(f"{TEST_FOLDER_PATH}/sample_file_{i}.txt")
        output_file.write_text(f"file{i}")

    return str(TEST_FOLDER_PATH)


def delete_local_folder(path):
    shutil.rmtree(path)


def test_github_folder():
    # TODO gh_folder = rh.folder(url='https://github.com/pytorch/pytorch', fs='github')
    gh_folder = rh.folder(
        url="/", fs="github", data_config={"org": "pytorch", "repo": "pytorch"}
    )
    assert gh_folder.ls()


# ----------------- Run tests -----------------


def test_from_cluster():
    cluster = rh.cluster(name="^rh-cpu").up_if_not()
    rh.folder(url=str(Path.cwd())).to(cluster, url="~/my_new_tests_folder")
    tests_folder = rh.folder(fs="file", url="~/my_new_tests_folder").from_cluster(
        cluster
    )
    assert "my_new_tests_folder/test_folder.py" in tests_folder.ls()


def test_create_and_save_data_to_s3_folder():
    data = list(range(50))
    s3_folder = rh.folder(url=DATA_STORE_PATH, fs="s3")
    s3_folder.mkdir()
    s3_folder.put({TEMP_FILE: pickle.dumps(data)}, overwrite=True)

    assert s3_folder.exists_in_fs()


def test_read_data_from_existing_s3_folder():
    # Note: Uses folder created above
    s3_folder = rh.folder(url=DATA_STORE_PATH, fs="s3")
    fss_file: "fsspec.core.OpenFile" = s3_folder.open(name=TEMP_FILE)
    with fss_file as f:
        data = pickle.load(f)

    assert data == list(range(50))


def test_create_and_delete_folder_from_s3():
    s3_folder = rh.folder(name=DATA_STORE_PATH, fs="s3", dryrun=False)
    s3_folder.mkdir()

    s3_folder.delete_configs()

    assert not s3_folder.exists_in_fs()


def test_cluster_tos():
    tests_folder = rh.folder(url=str(Path.cwd()))

    c = rh.cluster("^rh-cpu").up_if_not()
    # TODO [DG] change default behavior to return the from_cluster folder
    tests_folder = tests_folder.to(fs=c).from_cluster(c)
    assert "test_folder.py" in tests_folder.ls(full_paths=False)

    # to local
    local = tests_folder.to("here", url=TEST_FOLDER_PATH)
    assert "test_folder.py" in local.ls(full_paths=False)

    # to s3
    s3 = tests_folder.to("s3")
    assert "test_folder.py" in s3.ls(full_paths=False)

    delete_local_folder(TEST_FOLDER_PATH)
    s3.empty_folder()

    # to gcs
    gcs = tests_folder.to("gs")
    try:
        assert "test_folder.py" in gcs.ls(full_paths=False)
        gcs.empty_folder()

    except:
        bucket_name = gcs.bucket_name_from_url(gcs.url)
        print(
            f"Permissions to gs bucket ({bucket_name}) may not be fully enabled "
            f"on the cluster {c.name}. For now please manually enable them directly on the cluster. "
            f"See https://cloud.google.com/sdk/gcloud/reference/auth/login"
        )


def test_local_and_cluster():
    # Local to cluster
    local_folder = rh.folder(url=TEST_FOLDER_PATH)
    c = rh.cluster("^rh-cpu").up_if_not()
    cluster_folder = local_folder.to(fs=c).from_cluster(c)
    assert "sample_file_0.txt" in cluster_folder.ls(full_paths=False)

    # Cluster to local
    tmp_path = Path.cwd() / "tmp_from_cluster"
    local_from_cluster = cluster_folder.to("here", url=tmp_path)
    assert "sample_file_0.txt" in local_from_cluster.ls(full_paths=False)

    delete_local_folder(tmp_path)


def test_local_and_s3():
    # Local to S3
    local_folder = rh.folder(url=TEST_FOLDER_PATH)
    s3_folder = local_folder.to(fs="s3")
    assert "sample_file_0.txt" in s3_folder.ls(full_paths=False)

    # S3 to local
    tmp_path = Path.cwd() / "tmp_from_s3"
    tmp_path.mkdir(exist_ok=True)

    local_from_s3 = s3_folder.to("here", url=tmp_path)
    assert "sample_file_0.txt" in local_from_s3.ls(full_paths=False)

    delete_local_folder(tmp_path)
    s3_folder.empty_folder()


def test_local_and_gcs():
    # Local to GCS
    local_folder = rh.folder(url=TEST_FOLDER_PATH)
    gcs_folder = local_folder.to(fs="gs")
    assert "sample_file_0.txt" in gcs_folder.ls(full_paths=False)

    # GCS to local
    tmp_path = Path.cwd() / "tmp_from_gcs"
    tmp_path.mkdir(exist_ok=True)

    local_from_gcs = gcs_folder.to("here", url=tmp_path)
    assert "sample_file_0.txt" in local_from_gcs.ls(full_paths=False)

    delete_local_folder(tmp_path)
    gcs_folder.empty_folder()


def test_cluster_and_s3():
    # Local to cluster
    local_folder = rh.folder(url=TEST_FOLDER_PATH)
    c = rh.cluster("^rh-cpu").up_if_not()
    cluster_folder = local_folder.to(fs=c).from_cluster(c)
    assert "sample_file_0.txt" in cluster_folder.ls(full_paths=False)

    # Cluster to S3
    s3_folder = cluster_folder.to(fs="s3")
    assert "sample_file_0.txt" in s3_folder.ls(full_paths=False)

    # S3 to cluster
    cluster_from_s3 = s3_folder.to(fs=c)
    assert "sample_file_0.txt" in cluster_from_s3.ls(full_paths=False)

    s3_folder.empty_folder()


def test_cluster_and_gcs():
    # Local to cluster
    local_folder = rh.folder(url=TEST_FOLDER_PATH)
    c = rh.cluster("^rh-cpu").up_if_not()

    # Make sure we have gsutil and gcloud on the cluster - needed for copying the package + authenticating
    c.pip_install_packages(packages=["gsutil"])

    # TODO [JL] might be necessary to install gcloud on the cluster
    # c.run(['sudo snap install google-cloud-cli --classic'])

    cluster_folder = local_folder.to(fs=c).from_cluster(c)
    assert "sample_file_0.txt" in cluster_folder.ls(full_paths=False)

    # Cluster to GCS
    gcs_folder = cluster_folder.to(fs="gs")
    try:
        assert "sample_file_0.txt" in gcs_folder.ls(full_paths=False)
        # GCS to cluster
        cluster_from_gcs = gcs_folder.to(fs=c)
        assert "sample_file_0.txt" in cluster_from_gcs.ls(full_paths=False)

        gcs_folder.empty_folder()

    except:
        # TODO [JL] automate gcloud access on the cluster for writing to GCS bucket
        bucket_name = gcs_folder.bucket_name_from_url(gcs_folder.url)
        raise PermissionError(
            f"Permissions to gs bucket ({bucket_name}) may not be fully enabled "
            f"on the cluster {c.name}. For now please manually enable them directly on the cluster. "
            f"See https://cloud.google.com/sdk/gcloud/reference/auth/login"
        )


def test_cluster_and_cluster():
    # Local to cluster 1
    local_folder = rh.folder(url=TEST_FOLDER_PATH)
    c1 = rh.cluster("^rh-cpu").up_if_not()

    # Upload sky secrets to cluster - required when syncing over the folder from c1 to c2
    c1.send_secrets(providers=["sky"])

    cluster_folder_1 = local_folder.to(fs=c1).from_cluster(c1)
    assert "sample_file_0.txt" in cluster_folder_1.ls(full_paths=False)

    # Cluster 1 to cluster 2
    c2 = rh.cluster("^rh-8-cpu").up_if_not()
    cluster_folder_2 = cluster_folder_1.to(
        fs=c2, url=cluster_folder_1.url
    ).from_cluster(c2)
    assert "sample_file_0.txt" in cluster_folder_2.ls(full_paths=False)


def test_s3_and_s3():
    # Local to S3
    local_folder = rh.folder(url=TEST_FOLDER_PATH)
    s3_folder = local_folder.to(fs="s3")
    assert "sample_file_0.txt" in s3_folder.ls(full_paths=False)

    # from one s3 folder to another s3 folder
    new_s3_folder = s3_folder.to(fs="s3")
    assert "sample_file_0.txt" in new_s3_folder.ls(full_paths=False)

    s3_folder.empty_folder()
    new_s3_folder.empty_folder()


def test_gcs_and_gcs():
    # Local to GCS
    local_folder = rh.folder(url=TEST_FOLDER_PATH)
    gcs_folder = local_folder.to(fs="gs")
    assert "sample_file_0.txt" in gcs_folder.ls(full_paths=False)

    # from one gcs folder to another gcs folder
    new_gcs_folder = gcs_folder.to(fs="gs")
    assert "sample_file_0.txt" in new_gcs_folder.ls(full_paths=False)

    gcs_folder.empty_folder()
    new_gcs_folder.empty_folder()


def test_s3_and_gcs():
    # Local to S3
    local_folder = rh.folder(url=TEST_FOLDER_PATH)
    s3_folder = local_folder.to(fs="s3")
    assert "sample_file_0.txt" in s3_folder.ls(full_paths=False)

    # *** NOTE: transfers between providers are only supported at the bucket level at the moment (not directory) ***

    # S3 to GCS
    s3_folder_to_gcs = s3_folder.to(fs="gs")
    assert s3_folder_to_gcs.ls(full_paths=False)

    s3_folder.empty_folder()


def test_gcs_and_s3():
    # Local to GCS
    local_folder = rh.folder(url=TEST_FOLDER_PATH)
    gcs_folder = local_folder.to(fs="gs")
    assert "sample_file_0.txt" in gcs_folder.ls(full_paths=False)

    # *** NOTE: transfers between providers are only supported at the bucket level at the moment (not directory) ***

    # GCS to S3
    gcs_folder_to_s3 = gcs_folder.to(fs="s3")
    assert gcs_folder_to_s3.ls(full_paths=False)

    gcs_folder.empty_folder()


def test_s3_folder_uploads_and_downloads():
    # NOTE: you can specify a specific URL like this:
    # test_folder = rh.folder(url='/runhouse/my-folder', fs='gs')

    test_folder = rh.folder(fs="s3")
    test_folder.upload(src=TEST_FOLDER_PATH)

    assert test_folder.exists_in_fs()

    downloaded_url_folder = str(Path.cwd() / "downloaded_s3")
    test_folder.download(dest=downloaded_url_folder)

    assert Path(downloaded_url_folder).exists()

    test_folder.delete_in_fs()
    assert not test_folder.exists_in_fs()


if __name__ == "__main__":
    setup()
    unittest.main()
