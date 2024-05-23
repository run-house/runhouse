from pathlib import Path

import pytest

import runhouse as rh

import tests.test_resources.test_resource


class TestPackage(tests.test_resources.test_resource.TestResource):
    # TODO: torch extra index url on cluster, s3 / file packages

    MAP_FIXTURES = {"resource": "package"}

    packages = [
        "pip_package",
        "conda_package",
        "reqs_package",
        "local_package",
        "git_package",
    ]

    UNIT = {
        "package": packages,
    }

    LOCAL = {
        "package": packages,
        "cluster": ["docker_cluster_pk_ssh_no_auth"],
    }

    MINIMAL = {
        "package": packages,
        "cluster": ["ondemand_aws_cluster"],
    }

    RELEASE = {
        "package": packages,
        "cluster": ["ondemand_aws_cluster"],
    }

    @pytest.mark.level("unit")
    def test_package_factory_and_properties(self, package):
        assert isinstance(package, rh.Package)
        assert package.install_method in ["pip", "conda", "reqs", "local"]

    @pytest.mark.level("unit")
    @pytest.mark.parametrize(
        "pkg_str",
        [
            "numpy",
            "pip:numpy",
            "conda:numpy" "requirements.txt",
            "reqs:./",
            "local:./",
            "pip:https://github.com/runhouse/runhouse.git",
        ],
    )
    def test_from_string(self, pkg_str):
        package = rh.Package.from_string(pkg_str)
        assert isinstance(package, rh.Package)
        assert package.install_method in ["pip", "conda", "reqs", "local"]

        if package.install_method in ["reqs", "local"]:
            assert isinstance(package.install_target, rh.Folder)

    # --------- test install command ---------
    @pytest.mark.level("unit")
    def test_pip_install_cmd(self, pip_package):
        assert pip_package._install_cmd() == f"pip install {pip_package.install_target}"

    @pytest.mark.level("unit")
    def test_conda_install_cmd(self, conda_package):
        assert (
            conda_package._install_cmd()
            == f"conda install -y {conda_package.install_target}"
        )

    @pytest.mark.level("unit")
    def test_reqs_install_cmd(self, reqs_package):
        assert (
            reqs_package._install_cmd()
            == f"pip install -r {reqs_package.install_target.local_path}/requirements.txt"
        )

    @pytest.mark.level("unit")
    def test_git_install_cmd(self, git_package):
        assert git_package._install_cmd() == f"pip install {git_package.install_target}"

    # --------- test install on cluster ---------
    @pytest.mark.level("local")
    def test_pip_install(self, cluster, pip_package):
        assert (
            pip_package._install_cmd(cluster)
            == f"pip install {pip_package.install_target}"
        )

        # install through remote ssh
        pip_package._install(cluster=cluster)

        # install from on the cluster
        remote_package = cluster.put_resource(pip_package)
        cluster.call(remote_package, "_install")

    @pytest.mark.level("release")
    def test_conda_install(self, cluster, conda_package):
        assert (
            conda_package._install_cmd(cluster)
            == f"conda install -y {conda_package.install_target}"
        )

        # install through remote ssh
        conda_package._install(cluster=cluster)

        # install from on the cluster
        remote_package = cluster.put_resource(conda_package)
        cluster.call(remote_package, "_install")

    @pytest.mark.level("local")
    def test_remote_reqs_install(self, cluster, reqs_package):
        path = reqs_package.to(cluster).install_target.path

        assert reqs_package._install_cmd(cluster=cluster) in [
            None,
            f"pip install -r {path}/requirements.txt",
        ]
        reqs_package._install(cluster=cluster)

    @pytest.mark.level("local")
    def test_git_install(self, cluster, git_package):
        git_package._install(cluster=cluster)

    @pytest.mark.level("local")
    def test_local_reqs_on_cluster(self, cluster, local_package):
        remote_package = local_package.to(cluster)

        assert isinstance(remote_package.install_target, rh.Folder)
        assert remote_package.install_target.system == cluster

    # --------- basic torch index-url testing ---------
    @pytest.mark.level("unit")
    def test_torch_pip_install_command(self):
        pkg = rh.Package.from_string("torch")
        assert (
            pkg._install_cmd_for_torch("torch")
            == f"torch --index-url {rh.Package.TORCH_INDEX_URLS.get('cpu')} --extra-index-url https://pypi.python.org/simple/"
        )

    @pytest.mark.level("unit")
    def test_torch_reqs_install_command(self):
        reqs_lines = ["torch", "accelerate"]
        test_reqs_file = Path(__file__).parent / "requirements.txt"
        with open(test_reqs_file, "w") as f:
            f.writelines([line + "\n" for line in reqs_lines])

        dummy_pkg = rh.Package.from_string(specifier="pip:dummy_package")
        assert (
            dummy_pkg._requirements_txt_install_cmd(test_reqs_file, reqs_lines)
            == f"-r {test_reqs_file} --extra-index-url {rh.Package.TORCH_INDEX_URLS.get('cpu')}"
        )

        test_reqs_file.unlink()
