import subprocess
import unittest
from pathlib import Path

import pytest

import runhouse as rh
from runhouse import rh_config

extra_index_url = "--extra-index-url https://pypi.python.org/simple/"
cuda_116_url = "--index-url https://download.pytorch.org/whl/cu116"


def _create_s3_package():
    import shutil

    from runhouse.rns.api_utils.utils import create_s3_bucket

    s3_bucket_path = "runhouse-folder"
    folder_name = "tmp_s3_package"

    create_s3_bucket(s3_bucket_path)
    # Create a local temp folder to install for the package
    tmp_path = Path(rh_config.rns_client.locate_working_dir()) / folder_name
    tmp_path.mkdir(parents=True, exist_ok=True)
    for i in range(3):
        output_file = Path(f"{tmp_path}/sample_file_{i}.txt")
        output_file.write_text(f"file{i}")

    pkg = rh.Package.from_string(f"local:./{folder_name}")
    s3_pkg = pkg.to(system="s3", path=f"/{s3_bucket_path}/package-tests")

    shutil.rmtree(tmp_path)
    return s3_pkg, folder_name


def setup():
    pass


def summer(a, b):
    return a + b


def send_tensor_to_cuda():
    try:
        import torch

        a = torch.LongTensor(1).random_(0, 10)
        a = a.to(device="cuda")
        return isinstance(a, torch.Tensor)

    except Exception as e:
        return e


@pytest.mark.localtest
def test_from_string():
    p = rh.Package.from_string("reqs:~/runhouse")
    if (Path.home() / "runhouse").exists():
        assert isinstance(p.install_target, rh.Folder)
        assert p.install_target.path == str(Path.home() / "runhouse")
    else:
        assert p.install_target == "~/runhouse"


@pytest.mark.clustertest
@pytest.mark.rnstest
def test_share_package(cpu_cluster):
    import shutil

    # Create a local temp folder to install for the package
    tmp_path = Path.cwd().parent / "tmp_package"
    tmp_path.mkdir(parents=True, exist_ok=True)
    for i in range(3):
        output_file = Path(f"{tmp_path}/sample_file_{i}.txt")
        output_file.write_text(f"file{i}")

    p = rh.Package.from_string("local:./tmp_package")
    p.name = "package_to_share"  # shareable resource requires a name

    p.to(system=cpu_cluster)

    p.share(
        users=["josh@run.house", "donny@run.house"],
        access_type="write",
        notify_users=False,
    )

    shutil.rmtree(tmp_path)

    # Confirm the package's folder is now on the cluster
    status_codes = cpu_cluster.run(commands=["ls tmp_package"])
    assert "sample_file_0.txt" in status_codes[0][1]


@pytest.mark.rnstest
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


@pytest.mark.rnstest
def test_load_shared_git_package():
    git_package = rh.Package.from_name(name="@/shared_git_package")
    assert git_package.config_for_rns


@pytest.mark.clustertest
def test_local_package_function(cpu_cluster):
    function = rh.function(fn=summer).to(cpu_cluster, env=["./"])

    req = function.env.reqs[0]
    assert isinstance(req, rh.Package)
    assert isinstance(req.install_target, rh.Folder)
    assert req.install_target.system == cpu_cluster


@pytest.mark.clustertest
def test_local_package_to_cluster(cpu_cluster):
    package = rh.Package.from_string("./").to(cpu_cluster)

    assert isinstance(package.install_target, rh.Folder)
    assert package.install_target.system == cpu_cluster


@pytest.mark.clustertest
def test_mount_local_package_to_cluster(cpu_cluster):
    mount_path = "package_mount"
    package = rh.Package.from_string("./").to(cpu_cluster, path=mount_path, mount=True)

    assert isinstance(package.install_target, rh.Folder)
    assert package.install_target.system == cpu_cluster
    assert mount_path in cpu_cluster.run(["ls"])[0][1]


@pytest.mark.clustertest
@pytest.mark.awstest
def test_package_file_system_to_cluster(cpu_cluster):
    import shutil
    s3_pkg, folder_name = _create_s3_package()

    assert s3_pkg.install_target.system == "s3"
    assert s3_pkg.install_target.exists_in_system()

    s3_pkg.to(system=cpu_cluster, mount=True, path=folder_name)

    # Confirm the package's folder is now on the cluster
    assert "sample_file_0.txt" in cpu_cluster.run([f"ls {folder_name}"])[0][1]


@pytest.mark.localtest
def test_torch_install_command_generator_from_reqs():
    """For a given list of packages as listed in a requirements.txt file, modify them to include the full
    install commands (without running on the actual cluster)"""
    test_reqs_file = Path(__file__).parent / "requirements.txt"

    # [Required as listed in reqs.txt, expected formatted install cmd]
    packages_to_install = [
        [f"torch {extra_index_url}", f"torch {cuda_116_url} {extra_index_url}"],
        ["torchaudio", f"torchaudio {cuda_116_url} {extra_index_url}"],
        ["matchatorch", f"matchatorch {cuda_116_url} {extra_index_url}"],
        ["torch==1.13.0", f"torch==1.13.0 {cuda_116_url} {extra_index_url}"],
        [
            f"torch {cuda_116_url} {extra_index_url}",
            f"torch {cuda_116_url} {extra_index_url}",
        ],
        [
            f"torch>=1.13.0 {cuda_116_url}",
            f"torch>=1.13.0 {cuda_116_url} {extra_index_url}",
        ],
    ]

    # Write these packages to a temp reqs file, then call the function which creates the full install command
    package_names = [p[0] for p in packages_to_install]
    with open(test_reqs_file, "w") as f:
        f.writelines([line + "\n" for line in package_names])

    dummy_pkg = rh.Package.from_string(specifier="pip:dummy_package")

    reqs_from_file: list = dummy_pkg.format_torch_cmd_in_reqs_file(
        path=test_reqs_file, cuda_version_or_cpu="11.6"
    )

    for idx, install_cmd in enumerate(reqs_from_file):
        assert install_cmd == packages_to_install[idx][1]

    test_reqs_file.unlink()

    assert True


@pytest.mark.localtest
def test_torch_install_command_generator():
    """Checks that the command itself is correct (without actually running it on the cluster)"""
    cuda_version = "11.6"
    # [Mock command received by user, cuda version, expected formatted command]
    packages_to_install = [
        [
            f"torch==1.13.1 {extra_index_url}",
            cuda_version,
            f"torch==1.13.1 {cuda_116_url} {extra_index_url}",
        ],
        [
            f"torch {cuda_116_url} {extra_index_url}",
            cuda_version,
            f"torch {cuda_116_url} {extra_index_url}",
        ],
        ["matchatorch", cuda_version, f"matchatorch {cuda_116_url} {extra_index_url}"],
        [
            "torch>=1.13.0, <2.0.0",
            cuda_version,
            f"torch>=1.13.0,<2.0.0 {cuda_116_url} {extra_index_url}",
        ],
        [
            f"torch {cuda_116_url} torchaudio {extra_index_url}",
            cuda_version,
            f"torch {cuda_116_url} {extra_index_url} "
            f"torchaudio {cuda_116_url} "
            f"{extra_index_url}",
        ],
        [
            "torch torchpudding",
            cuda_version,
            f"torch {cuda_116_url} {extra_index_url} torchpudding {cuda_116_url} {extra_index_url}",
        ],
        [
            "torch>=1.13.0",
            cuda_version,
            f"torch>=1.13.0 {cuda_116_url} {extra_index_url}",
        ],
        [
            f"torch>1.13.0 {extra_index_url}",
            cuda_version,
            f"torch>1.13.0 {cuda_116_url} {extra_index_url}",
        ],
        [
            f"torch==1.13.0 {cuda_116_url}",
            cuda_version,
            f"torch==1.13.0 {cuda_116_url} {extra_index_url}",
        ],
        [
            f"torch==1.13.0 {cuda_116_url}",
            cuda_version,
            f"torch==1.13.0 {cuda_116_url} {extra_index_url}",
        ],
        [
            "torch~=1.13.0",
            cuda_version,
            f"torch~=1.13.0 {cuda_116_url} {extra_index_url}",
        ],
        [
            f"torch==99.99.999 {cuda_116_url}",
            cuda_version,
            f"torch==99.99.999 {cuda_116_url} {extra_index_url}",
        ],
    ]
    for cmds in packages_to_install:
        torch_version, cuda_version, expected_install_cmd = cmds
        dummy_pkg = rh.Package.from_string(specifier=f"pip:{torch_version}")
        formatted_install_cmd = dummy_pkg.install_cmd_for_torch(
            torch_version, cuda_version
        )

        assert (
            formatted_install_cmd == expected_install_cmd
        ), f"Unexpected result for {(torch_version, cuda_version)} "


@pytest.mark.gputest
@pytest.mark.parametrize(
    "cluster",
    ["cpu_cluster", "v100_gpu_cluster", "k80_gpu_cluster", "a10g_gpu_cluster"],
    indirect=True,
)
def test_getting_cuda_version_on_clusters(request, cluster):
    """Gets the cuda version on the cluster and asserts it is the expected version"""
    return_codes: list = cluster.run_python(
        ["import runhouse as rh", "print(rh.Package.detect_cuda_version_or_cpu())"]
    )
    cuda_version_or_cpu = return_codes[0][1].strip().split("\n")[-1]
    print(f"{cluster.name}: {cuda_version_or_cpu}")

    instance_type = cluster.instance_type.lower()

    # Save the cuda version (or indicate cpu) for each cluster
    request.config.cache.set(instance_type, cuda_version_or_cpu)

    if instance_type.startswith("k80"):
        assert cuda_version_or_cpu == "11.3"
    elif instance_type.startswith("cpu"):
        assert cuda_version_or_cpu == "cpu"
    else:
        assert cuda_version_or_cpu == "11.7"


@pytest.mark.gputest
@pytest.mark.parametrize(
    "cluster",
    ["cpu_cluster", "v100_gpu_cluster", "k80_gpu_cluster", "a10g_gpu_cluster"],
    indirect=True,
)
def test_install_cmd_for_torch_on_cluster(request, cluster):
    """Checks that the install command for torch runs properly on the cluster.
    Confirms that we can properly install the package (and send a torch tensor to cuda to validate it"""
    # Use the cuda version on the cluster that we got from the previous test
    cuda_version_or_cpu = request.config.cache.get(cluster.instance_type.lower(), None)

    assert "AttributeError" not in cuda_version_or_cpu, (
        f"No cuda version saved for {cluster.name} - run `test_getting_cuda_version_on_clusters`"
        "to save the cuda version"
    )

    cuda_url = rh.Package.TORCH_INDEX_URLS.get(cuda_version_or_cpu)
    cuda_index_url = f"--index-url {cuda_url}"

    install_commands_for_cluster = [
        f"torch==1.13.1 {cuda_index_url} {extra_index_url}",
        f"torch>=1.13.0 {extra_index_url}",
        f"torch>1.13.0 {cuda_index_url}",
        f"torch {cuda_index_url}",
        f"torch>=1.13.0,<=2.0.0 {cuda_index_url}",
        f"torch==2.0.0 {cuda_index_url} torchaudio {cuda_index_url}",
        f"torch~=2.0.0 {cuda_index_url}",
    ]

    for install_cmd in install_commands_for_cluster:
        # Run the complete install command on the cluster
        try:
            cluster.install_packages([install_cmd])
        except subprocess.CalledProcessError:
            assert False, f"Failed to install {install_cmd}"

    if cuda_version_or_cpu != "cpu":
        # Send a tensor to CUDA using the installed torch package (ignore if we are on CPU)
        tensor_to_cuda = rh.function(fn=send_tensor_to_cuda).to(
            cluster, reqs=["pytest"]
        )
        sent_to_cuda = tensor_to_cuda()
        assert sent_to_cuda, f"Failed to send torch tensor to CUDA on {cluster.name}"


if __name__ == "__main__":
    setup()
    unittest.main()
