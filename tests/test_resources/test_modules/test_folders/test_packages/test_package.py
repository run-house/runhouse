import subprocess
from pathlib import Path

import pytest

import runhouse as rh

extra_index_url = "--extra-index-url https://pypi.python.org/simple/"
cuda_116_url = "--index-url https://download.pytorch.org/whl/cu116"


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


def test_from_string():
    p = rh.Package.from_string("reqs:~/runhouse")
    if (Path.home() / "runhouse").exists():
        assert isinstance(p.install_target, rh.Folder)
        assert p.install_target.path == str(Path.home() / "runhouse")
    else:
        assert p.install_target == "~/runhouse"


def test_share_package(ondemand_aws_cluster, local_package):
    local_package.to(system=ondemand_aws_cluster)
    local_package.save("package_to_share")  # shareable resource requires a name

    local_package.share(
        users=["josh@run.house", "donny@run.house"],
        access_level="write",
        notify_users=False,
    )

    # TODO test loading from a different account for real
    # Confirm the package's folder is now on the cluster
    status_codes = ondemand_aws_cluster.run(commands=["ls tmp_package"])
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
        access_level="read",
        notify_users=False,
    )
    assert rh.exists(name="shared_git_package")


def test_load_shared_git_package():
    git_package = rh.package(name="@/shared_git_package")
    assert git_package.config()


def test_local_package_function(cluster):
    function = rh.function(fn=summer).to(cluster, env=["./"])

    req = function.env.reqs[0]
    assert isinstance(req, rh.Package)
    assert isinstance(req.install_target, rh.Folder)
    assert req.install_target.system == cluster


def test_local_package_to_cluster(cluster):
    package = rh.Package.from_string("./").to(cluster)

    assert isinstance(package.install_target, rh.Folder)
    assert package.install_target.system == cluster


def test_mount_local_package_to_cluster(cluster):
    mount_path = "package_mount"
    package = rh.Package.from_string("./").to(cluster, path=mount_path, mount=True)

    assert isinstance(package.install_target, rh.Folder)
    assert package.install_target.system == cluster
    assert mount_path in cluster.run(["ls"])[0][1]


def test_package_file_system_to_cluster(cluster, s3_package):
    assert s3_package.install_target.system == "s3"
    assert s3_package.install_target.exists_in_system()

    folder_name = Path(s3_package.install_target.path).stem
    s3_package.to(system=cluster, mount=True, path=folder_name)

    # Confirm the package's folder is now on the cluster
    assert "sample_file_0.txt" in cluster.run([f"ls {folder_name}"])[0][1]


@pytest.mark.parametrize(
    "reqs_lines",
    [
        [cuda_116_url, "", "torch"],
        ["diffusers", "accelerate"],
        [f"torch {cuda_116_url}"],
    ],
)
def test_basic_command_generator_from_reqs(reqs_lines):
    test_reqs_file = Path(__file__).parent / "requirements.txt"
    with open(test_reqs_file, "w") as f:
        f.writelines([line + "\n" for line in reqs_lines])

    dummy_pkg = rh.Package.from_string(specifier="pip:dummy_package")
    install_cmd = dummy_pkg._requirements_txt_install_cmd(test_reqs_file)

    assert install_cmd == f"-r {test_reqs_file}"

    test_reqs_file.unlink()
    assert True


def test_command_generator_from_reqs():
    reqs_lines = ["torch", "accelerate"]
    test_reqs_file = Path(__file__).parent / "requirements.txt"
    with open(test_reqs_file, "w") as f:
        f.writelines([line + "\n" for line in reqs_lines])

    dummy_pkg = rh.Package.from_string(specifier="pip:dummy_package")
    install_cmd = dummy_pkg._requirements_txt_install_cmd(
        test_reqs_file, cuda_version_or_cpu="11.6"
    )

    assert (
        install_cmd
        == f"-r {test_reqs_file} --extra-index-url https://download.pytorch.org/whl/cu116"
    )

    test_reqs_file.unlink()
    assert True


def test_torch_install_command_generator_from_reqs():
    """Test correctly generating full install commands for torch-related packages."""
    test_cuda_version = "11.6"

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

    dummy_pkg = rh.Package.from_string(specifier="pip:dummy_package")
    reformatted_packaged_to_install = [
        dummy_pkg._install_cmd_for_torch(p[0], test_cuda_version)
        for p in packages_to_install
    ]

    for idx, install_cmd in enumerate(reformatted_packaged_to_install):
        assert install_cmd == packages_to_install[idx][1]


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
        formatted_install_cmd = dummy_pkg._install_cmd_for_torch(
            torch_version, cuda_version
        )

        assert (
            formatted_install_cmd == expected_install_cmd
        ), f"Unexpected result for {(torch_version, cuda_version)} "


def test_getting_cuda_version_on_clusters(request, ondemand_cluster):
    """Gets the cuda version on the cluster and asserts it is the expected version"""
    return_codes: list = ondemand_cluster.run_python(
        ["import runhouse as rh", "print(rh.Package._detect_cuda_version_or_cpu())"]
    )
    cuda_version_or_cpu = return_codes[0][1].strip().split("\n")[-1]
    print(f"{ondemand_cluster.name}: {cuda_version_or_cpu}")

    instance_type = ondemand_cluster.instance_type.lower()

    # Save the cuda version (or indicate cpu) for each cluster
    request.config.cache.set(instance_type, cuda_version_or_cpu)

    if instance_type.startswith("k80"):
        assert cuda_version_or_cpu == "11.3"
    elif instance_type.startswith("cpu"):
        assert cuda_version_or_cpu == "cpu"
    else:
        assert cuda_version_or_cpu == "11.7"


def test_install_cmd_for_torch_on_cluster(request, ondemand_cluster):
    """Checks that the install command for torch runs properly on the cluster.
    Confirms that we can properly install the package (and send a torch tensor to cuda to validate it"""
    # Use the cuda version on the cluster that we got from the previous test
    cuda_version_or_cpu = request.config.cache.get(
        ondemand_cluster.instance_type.lower(), None
    )

    assert "AttributeError" not in cuda_version_or_cpu, (
        f"No cuda version saved for {ondemand_cluster.name} - run `test_getting_cuda_version_on_clusters`"
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
            ondemand_cluster.install_packages([install_cmd])
        except subprocess.CalledProcessError:
            assert False, f"Failed to install {install_cmd}"

    if cuda_version_or_cpu != "cpu":
        # Send a tensor to CUDA using the installed torch package (ignore if we are on CPU)
        tensor_to_cuda = rh.function(fn=send_tensor_to_cuda).to(
            ondemand_cluster, reqs=["pytest"]
        )
        sent_to_cuda = tensor_to_cuda()
        assert (
            sent_to_cuda
        ), f"Failed to send torch tensor to CUDA on {ondemand_cluster.name}"
