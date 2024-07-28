# Package tests mostly migrated to tests/test_resources/test_data/test_package.py

import subprocess
from pathlib import Path

import runhouse as rh

extra_index_url = "--extra-index-url https://pypi.python.org/simple/"
cuda_116_url = "--index-url https://download.pytorch.org/whl/cu116"


def send_tensor_to_cuda():
    try:
        import torch

        a = torch.LongTensor(1).random_(0, 10)
        a = a.to(device="cuda")
        return isinstance(a, torch.Tensor)

    except Exception as e:
        return e


def test_mount_local_package_to_cluster(cluster):
    mount_path = "package_mount"
    package = rh.Package.from_string("./").to(cluster, path=mount_path)

    assert isinstance(package.install_target, rh.Folder)
    assert package.install_target.system == cluster
    assert mount_path in cluster.run(["ls"])[0][1]


def test_package_file_system_to_cluster(cluster, s3_package):
    assert s3_package.install_target.system == "s3"
    assert s3_package.install_target.exists_in_system()

    folder_name = Path(s3_package.install_target.path).stem
    s3_package.to(system=cluster, path=folder_name)

    # Confirm the package's folder is now on the cluster
    assert "sample_file_0.txt" in cluster.run([f"ls {folder_name}"])[0][1]


def test_getting_cuda_version_on_clusters(request, ondemand_cluster):
    """Gets the cuda version on the cluster and asserts it is the expected version"""
    return_codes: list = ondemand_cluster.run_python(
        [
            "from runhouse.resources.hardware.utils import detect_cuda_version_or_cpu",
            "print(detect_cuda_version_or_cpu())",
        ]
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
