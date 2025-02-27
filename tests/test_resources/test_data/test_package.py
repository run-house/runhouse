import sys
from pathlib import Path

import pytest

import runhouse as rh
import tests.test_resources.test_resource
from runhouse.resources.packages import InstallTarget
from runhouse.resources.packages.package import INSTALL_METHODS
from runhouse.utils import run_with_logs


def get_bs4_version():
    import bs4

    return bs4.__version__


class TestPackage(tests.test_resources.test_resource.TestResource):
    # TODO: torch extra index url on cluster, s3 / file packages

    MAP_FIXTURES = {"resource": "package"}

    packages = [
        "pip_package",
        "conda_package",
        "local_package",
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
        "cluster": ["local_launched_ondemand_aws_docker_cluster"],
    }

    RELEASE = {
        "package": packages,
        "cluster": ["local_launched_ondemand_aws_docker_cluster"],
    }

    MAXIMAL = {
        "package": packages,
        "cluster": ["local_launched_ondemand_aws_docker_cluster"],
    }

    @pytest.mark.level("unit")
    def test_package_factory_and_properties(self, package):
        assert isinstance(package, rh.Package)
        assert package.install_method in INSTALL_METHODS

    @pytest.mark.level("unit")
    @pytest.mark.parametrize(
        "pkg_str",
        [
            "numpy",
            "pip:numpy",
            "conda:numpy",
            "local:./",
        ],
    )
    def test_from_string(self, pkg_str):
        package = rh.Package.from_string(pkg_str)
        assert isinstance(package, rh.Package)
        assert package.install_method in INSTALL_METHODS

        if package.install_method == "local":
            assert isinstance(package.install_target, InstallTarget)

    # --------- test install command ---------
    @pytest.mark.level("unit")
    def test_pip_install_cmd(self, pip_package):
        assert (
            pip_package._pip_install_cmd()
            == f'{sys.executable} -m pip install "{pip_package.install_target}"'
        )

    @pytest.mark.level("unit")
    def test_conda_install_cmd(self, conda_package):
        assert (
            conda_package._conda_install_cmd()
            == f"conda install -y {conda_package.install_target}"
        )

    # --------- test install on cluster ---------
    @pytest.mark.level("local")
    def test_pip_install(self, cluster, pip_package):
        assert (
            pip_package._pip_install_cmd(cluster=cluster)
            == f'python3 -m pip install "{pip_package.install_target}"'
        )

        # install through remote ssh
        pip_package._install(cluster=cluster)

        # install from on the cluster
        remote_package = cluster.put_resource(pip_package)
        cluster.call(remote_package, "_install")

    @pytest.mark.level("release")
    def test_conda_install(self, cluster, conda_package):
        assert (
            conda_package._conda_install_cmd(cluster=cluster)
            == f"conda install -y {conda_package.install_target}"
        )

        # install through remote ssh
        conda_package._install(cluster=cluster)

        # install from on the cluster
        remote_package = cluster.put_resource(conda_package)
        cluster.call(remote_package, "_install")

    @pytest.mark.level("local")
    def test_local_reqs_on_cluster(self, cluster, local_package):
        remote_package = local_package.to(cluster)

        assert isinstance(remote_package.install_target, InstallTarget)

    @pytest.mark.level("local")
    def test_local_package_version_gets_installed(self, cluster):
        run_with_logs("pip install beautifulsoup4==4.11.1")
        cluster.pip_install(["beautifulsoup4"])

        process = cluster.ensure_process_created("temp_env")
        remote_fn = rh.function(get_bs4_version).to(cluster, process=process)
        assert remote_fn() == "4.11.1"

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
            f"-r {test_reqs_file} --extra-index-url {rh.Package.TORCH_INDEX_URLS.get('cpu')}"
            in dummy_pkg._reqs_install_cmd_for_torch(test_reqs_file, reqs_lines)
        )

        test_reqs_file.unlink()

    @pytest.mark.level("local")
    def test_package_in_home_dir_to_cluster(self, cluster):
        with pytest.raises(rh.CodeSyncError):
            rh.Package.from_string("~").to(cluster)
