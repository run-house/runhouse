import unittest
from pathlib import Path

import runhouse as rh


def setup():
    pass


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


def test_torch_installs():
    cuda_url = "--index-url https://download.pytorch.org/whl/cu116"
    extra_index_url = "--extra-index-url https://pypi.python.org/simple/"

    # Torch install commands and their expected results to be pip installed
    install_commands = {
        "torch==1.13.1": "torch==1.13.1 --extra-index-url https://pypi.python.org/simple/",
        f"torch==1.13.1 {extra_index_url}": f"torch==1.13.1 {extra_index_url}",
        "matchatorch": "matchatorch --extra-index-url https://pypi.python.org/simple/",
        "torch>=1.13.0, <2.0.0": f"torch>=1.13.0, <2.0.0 {extra_index_url}",
        f"torch torchaudio {cuda_url}": f"torch {extra_index_url} torchaudio {cuda_url}",
        f"torchaudio {cuda_url} torch {cuda_url}": f"torchaudio {cuda_url} torch {cuda_url}",
        "torch torchpudding": f"torch {extra_index_url} torchpudding {extra_index_url}",
        f"torch==1.13.1 {cuda_url}": f"torch==1.13.1 {cuda_url}",
        "torch>=1.13.0": f"torch>=1.13.0 {extra_index_url}",
        "torch>1.13.0": f"torch>1.13.0 {extra_index_url}",
        "torch~=1.13.0": f"torch~=1.13.0 {extra_index_url}",
        "torch==99.99.999": f"torch==99.99.999 {extra_index_url}",
    }
    for mock_install_cmd, expected_install_cmd in install_commands.items():
        dummy_pkg = rh.Package.from_string(specifier=f"pip:{mock_install_cmd}")
        formatted_install_cmd = dummy_pkg.install_cmd_for_torch(mock_install_cmd)

        assert (
            formatted_install_cmd == expected_install_cmd
        ), f"Unexpected result for command {mock_install_cmd} "


def test_cuda_versions_for_hardware():
    cpu = rh.cluster("^rh-cpu").up_if_not()
    a10g = rh.cluster(
        name="rh-a10x", instance_type="g5.2xlarge", provider="aws"
    ).up_if_not()

    cuda_url_for_cluster = {
        cpu: "https://download.pytorch.org/whl/cu117",
        a10g: "https://download.pytorch.org/whl/cu117",
    }

    for cluster, expected_cuda_url in cuda_url_for_cluster.items():
        return_codes = cluster.run(["nvcc --version"], stream_logs=True)
        cuda_version = return_codes[0][1].split("release ")[1].split(",")[0]
        cuda_url = rh.Package.TORCH_INDEX_URLS_FOR_CUDA.get(cuda_version)
        assert cuda_url == expected_cuda_url


if __name__ == "__main__":
    setup()
    unittest.main()
