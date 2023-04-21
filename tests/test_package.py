import unittest
from pathlib import Path

import pytest

import runhouse as rh

CUDA_116_URL = "--index-url https://download.pytorch.org/whl/cu116"
CUDA_117_URL = "--index-url https://download.pytorch.org/whl/cu117"
CUDA_118_URL = "--index-url https://download.pytorch.org/whl/cu118"
EXTRA_INDEX_URL = "--extra-index-url https://pypi.python.org/simple/"

INSTALL_COMMANDS = {
    (
        f"torch==1.13.1 {CUDA_116_URL}",
        "11.6",
    ): f"torch==1.13.1 {CUDA_116_URL} {EXTRA_INDEX_URL}",
    (
        f"torch {EXTRA_INDEX_URL}",
        "11.7",
    ): f"torch {EXTRA_INDEX_URL} {CUDA_117_URL}",
    ("matchatorch", "11.6"): f"matchatorch {CUDA_116_URL} {EXTRA_INDEX_URL}",
    (
        "torch>=1.13.0, <2.0.0",
        "11.8",
    ): f"torch>=1.13.0, <2.0.0 {CUDA_118_URL} {EXTRA_INDEX_URL}",
    (
        f"torch {CUDA_116_URL} torchaudio {EXTRA_INDEX_URL}",
        "11.6",
    ): f"torch {CUDA_116_URL} {EXTRA_INDEX_URL} "
    f"torchaudio {EXTRA_INDEX_URL} "
    f"{CUDA_116_URL}",
    (
        "torch torchpudding",
        "11.7",
    ): f"torch {CUDA_117_URL} {EXTRA_INDEX_URL} torchpudding {CUDA_117_URL} "
    f"{EXTRA_INDEX_URL}",
    ("torch>=1.13.0", "11.6"): f"torch>=1.13.0 {CUDA_116_URL} {EXTRA_INDEX_URL}",
    ("torch>1.13.0", "11.6"): f"torch>1.13.0 {CUDA_116_URL} {EXTRA_INDEX_URL}",
    ("torch==1.13.0", "11.6"): f"torch==1.13.0 {CUDA_116_URL} {EXTRA_INDEX_URL}",
    ("torch~=1.13.0", "11.7"): f"torch~=1.13.0 {CUDA_117_URL} {EXTRA_INDEX_URL}",
    (
        f"torch==99.99.999 {CUDA_117_URL}",
        "11.7",
    ): f"torch==99.99.999 {CUDA_117_URL} {EXTRA_INDEX_URL}",
}


def setup():
    pass


@pytest.fixture
def cluster(request):
    return request.getfixturevalue(request.param)


def test_from_string():
    p = rh.Package.from_string("reqs:~/runhouse")
    assert isinstance(p.install_target, rh.Folder)
    assert p.install_target.path == str(Path.home() / "runhouse")


def test_share_package():
    import shutil

    # Create a local temp folder to install for the package
    tmp_path = Path.cwd().parent / "tmp_package"
    tmp_path.mkdir(parents=True, exist_ok=True)
    for i in range(3):
        output_file = Path(f"{tmp_path}/sample_file_{i}.txt")
        output_file.write_text(f"file{i}")

    p = rh.Package.from_string("local:./tmp_package")
    p.name = "package_to_share"  # shareable resource requires a name

    c = rh.cluster(name="@/rh-cpu")
    p.to_cluster(dest_cluster=c)

    p.share(
        users=["josh@run.house", "donny@run.house"],
        access_type="write",
        notify_users=False,
    )

    shutil.rmtree(tmp_path)

    # Confirm the package's folder is now on the cluster
    status_codes = c.run(commands=["ls tmp_package"])
    assert "sample_file_0.txt" in status_codes[0][1]


def test_share_git_package():
    git_package = rh.GitPackage(
        name="shared_git_package",
        git_url="https://github.com/runhouse/runhouse.git",
        install_method="pip",
        revision="v0.0.1",
    ).save()

    git_package.share(
        users=["donny@run.house", "josh@run.house"],
        access_type="read",
        notify_users=False,
    )
    assert rh.rns_client.exists(name_or_path="shared_git_package")


def test_load_shared_git_package():
    git_package = rh.Package.from_name(name="@/shared_git_package")
    assert git_package.config_for_rns


def test_install_command_for_torch_locally():
    for (torch_version, cuda_version), expected_install_cmd in INSTALL_COMMANDS.items():
        dummy_pkg = rh.Package.from_string(specifier=f"pip:{torch_version}")
        formatted_install_cmd = dummy_pkg.install_cmd_for_torch(
            torch_version, cuda_version
        )

        assert (
            formatted_install_cmd == expected_install_cmd
        ), f"Unexpected result for {(torch_version, cuda_version)} "


@pytest.mark.parametrize("cluster", ["cpu", "v100"], indirect=True)
def test_getting_cuda_version_on_clusters(cluster):
    return_codes: list = cluster.run_python(
        ["import runhouse as rh", "print(rh.Package.detect_cuda_version())"]
    )
    cuda_version_on_custer = return_codes[0][1].strip().split("\n")[-1]
    assert cuda_version_on_custer == "11.7"


@unittest.skip("Not implemented")
@pytest.mark.parametrize("cluster", ["cpu", "v100"], indirect=True)
def test_install_cmd_for_torch_on_cluster(cluster):
    # taking the install strings and create a package with that string, send it to each of the clusters and
    # try installing it (call install on the package)
    # Send a cluster.run() or define a function which gets the PyTorch version and sends a tensor to CUDA
    pass


if __name__ == "__main__":
    setup()
    unittest.main()
