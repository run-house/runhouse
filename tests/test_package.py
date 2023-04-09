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

    c = rh.cluster(name="^rh-cpu")
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
    simple_index_url = "--extra-index-url https://pypi.python.org/simple/"

    # Torch install commands and their expected results (None if the install command is invalid)
    install_commands = {
        "torch==1.13.1": "torch==1.13.1",
        "torch>=1.13.0, <2.0.0": "torch>=1.13.0, <2.0.0",
        f"torch torchaudio {cuda_url}": f"torch {simple_index_url} torchaudio {cuda_url}",
        f"torchaudio {cuda_url} torch {cuda_url}": f"torchaudio {cuda_url} torch {cuda_url}",
        "torch torchpudding": f"torch {simple_index_url}",
        "torch>=1.13.0": "torch>=1.13.0",
        "torch>1.13.0": "torch>1.13.0",
        "torch~=1.13.0": "torch~=1.13.0",
        "torch==1.13.1+cu118": "torch==1.13.1+cu118",
        "torchpudding": None,
        "torchpudding==1.13.1": None,
        "torch==99.99.999": None,
    }
    for install_cmd, expected_res in install_commands.items():
        dummy_pkg = rh.Package.from_string(specifier=f"pip:{install_cmd}")
        formatted_cmd = dummy_pkg.install_cmd_for_torch(install_cmd)
        assert (
            formatted_cmd == expected_res
        ), f"Unexpected response for command: {install_cmd}"


if __name__ == "__main__":
    setup()
    unittest.main()
