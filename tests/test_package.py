import unittest
from pathlib import Path

import pytest

import runhouse as rh

extra_index_url = "--extra-index-url https://pypi.python.org/simple/"


def setup():
    pass


@pytest.fixture
def cluster(request):
    return request.getfixturevalue(request.param)


def test_from_string():
    p = rh.Package.from_string("reqs:~/runhouse")
    assert isinstance(p.install_target, rh.Folder)
    assert p.install_target.path == str(Path.home() / "runhouse")


@pytest.mark.clustertest
def test_share_package(cpu):
    import shutil

    # Create a local temp folder to install for the package
    tmp_path = Path.cwd().parent / "tmp_package"
    tmp_path.mkdir(parents=True, exist_ok=True)
    for i in range(3):
        output_file = Path(f"{tmp_path}/sample_file_{i}.txt")
        output_file.write_text(f"file{i}")

    p = rh.Package.from_string("local:./tmp_package")
    p.name = "package_to_share"  # shareable resource requires a name

    p.to_cluster(dest_cluster=cpu)

    p.share(
        users=["josh@run.house", "donny@run.house"],
        access_type="write",
        notify_users=False,
    )

    shutil.rmtree(tmp_path)

    # Confirm the package's folder is now on the cluster
    status_codes = cpu.run(commands=["ls tmp_package"])
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
    """Checks that the command itself is correct (without actually running it on the cluster)"""
    cuda_116_url = "--index-url https://download.pytorch.org/whl/cu116"
    cuda_117_url = "--index-url https://download.pytorch.org/whl/cu117"
    cuda_118_url = "--index-url https://download.pytorch.org/whl/cu118"

    install_commands = {
        (
            f"torch==1.13.1 {cuda_116_url}",
            "11.6",
        ): f"torch==1.13.1 {cuda_116_url} {extra_index_url}",
        (
            f"torch {extra_index_url}",
            "11.7",
        ): f"torch {extra_index_url} {cuda_117_url}",
        ("matchatorch", "11.6"): f"matchatorch {cuda_116_url} {extra_index_url}",
        (
            "torch>=1.13.0, <2.0.0",
            "11.8",
        ): f"torch>=1.13.0,<2.0.0 {cuda_118_url} {extra_index_url}",
        (
            f"torch {cuda_116_url} torchaudio {extra_index_url}",
            "11.6",
        ): f"torch {cuda_116_url} {extra_index_url} "
        f"torchaudio {extra_index_url} "
        f"{cuda_116_url}",
        (
            "torch torchpudding",
            "11.7",
        ): f"torch {cuda_117_url} {extra_index_url} torchpudding {cuda_117_url} "
        f"{extra_index_url}",
        ("torch>=1.13.0", "11.6"): f"torch>=1.13.0 {cuda_116_url} {extra_index_url}",
        ("torch>1.13.0", "11.6"): f"torch>1.13.0 {cuda_116_url} {extra_index_url}",
        ("torch==1.13.0", "11.6"): f"torch==1.13.0 {cuda_116_url} {extra_index_url}",
        ("torch~=1.13.0", "11.7"): f"torch~=1.13.0 {cuda_117_url} {extra_index_url}",
        (
            f"torch==99.99.999 {cuda_117_url}",
            "11.7",
        ): f"torch==99.99.999 {cuda_117_url} {extra_index_url}",
    }
    for (torch_version, cuda_version), expected_install_cmd in install_commands.items():
        dummy_pkg = rh.Package.from_string(specifier=f"pip:{torch_version}")
        formatted_install_cmd = dummy_pkg.install_cmd_for_torch(
            torch_version, cuda_version
        )

        assert (
            formatted_install_cmd == expected_install_cmd
        ), f"Unexpected result for {(torch_version, cuda_version)} "


@pytest.mark.clustertest
@pytest.mark.parametrize("cluster", ["cpu", "v100", "k80"], indirect=True)
def test_getting_cuda_version_on_clusters(request, cluster):
    """Gets the cuda version on the cluster and asserts it is the expected version"""
    return_codes: list = cluster.run_python(
        ["import runhouse as rh", "print(rh.Package.detect_cuda_version())"]
    )
    cuda_version_on_cluster = return_codes[0][1].strip().split("\n")[-1]
    print(f"{cluster.name}: {cuda_version_on_cluster}")

    instance_type = cluster.instance_type.lower()
    request.config.cache.set(instance_type, cuda_version_on_cluster)

    if instance_type.startswith("k80"):
        assert cuda_version_on_cluster == "11.3"
    else:
        assert cuda_version_on_cluster == "11.7"


# TODO [JL] WIP
@pytest.mark.clustertest
@pytest.mark.parametrize("cluster", ["v100", "k80"], indirect=True)
def test_install_cmd_for_torch_on_cluster(request, cluster):
    """Checks that the install command for torch runs properly on the cluster.
    Confirms that we can properly install the package (and send a torch tensor to cuda to validate it"""
    # Get the cuda version on the cluster that we got from the previous test
    cuda_version_on_cluster = request.config.cache.get(
        cluster.instance_type.lower(), None
    )
    cuda_url = rh.Package.TORCH_INDEX_URLS_FOR_CUDA.get(cuda_version_on_cluster)
    cuda_index_url = f"--index-url {cuda_url}"

    install_commands_for_cluster = [
        f"torch==1.13.1 {cuda_index_url} {extra_index_url}",
        f"torch>1.13.0 {cuda_index_url} {extra_index_url}",
        f"torch {extra_index_url} {cuda_index_url}",
        f"torch>=1.13.0,<=2.0.0 {cuda_index_url} {extra_index_url}",
        f"torch==2.0.0 {cuda_index_url} {extra_index_url} torchaudio {cuda_index_url} {extra_index_url}",
        f"torch~=2.0.0 {cuda_index_url} {extra_index_url}",
    ]
    for install_cmd in install_commands_for_cluster:
        # Install the complete command on the cluster
        pip_install_cmd = cluster.run_python(
            [
                "import runhouse as rh",
                f"rh.Package.pip_install(install_cmd='{install_cmd}')",
            ]
        )
        assert (
            "ERROR" not in pip_install_cmd[0][1]
        ), f"Failed to install command on {cluster.name}"

        # Try sending a tensor to CUDA using the torch version installed on the cluster
        torch_cmds = cluster.run_python(
            [
                "import torch",
                "a = torch.LongTensor(1).random_(0, 10)",
                "a = a.to(device='cuda')",
                "print(a)",
            ]
        )
        assert (
            "RuntimeError" not in torch_cmds[0][1]
        ), f"Failed to send torch tensor to CUDA on cluster {cluster.name}"


if __name__ == "__main__":
    setup()
    unittest.main()
