import os

from pathlib import Path
from typing import Dict

import pytest

import runhouse as rh

import tests.test_resources.test_resource

from tests.conftest import init_args


def _get_env_var_value(env_var):
    import os

    return os.environ[env_var]


def _uninstall_env(env, cluster):
    for req in env.reqs:
        if "/" not in req:
            cluster.run([f"pip uninstall {req} -y"], env=env)


def np_summer(a, b):
    import numpy as np

    return int(np.sum([a, b]))


def scipy_import():
    import scipy

    return str(scipy.__version__)


@pytest.mark.envtest
class TestEnv(tests.test_resources.test_resource.TestResource):
    MAP_FIXTURES = {"resource": "env"}

    UNIT = {
        "env": [
            "unnamed_env",
            "named_env",
            "base_conda_env",
            "named_conda_env_from_dict",
            "conda_env_from_local",
            "conda_env_from_path",
        ]
    }
    LOCAL = {
        "env": ["unnamed_env", "named_env"],
        "cluster": [
            "docker_cluster_pk_ssh_no_auth",
        ]
        # TODO: extend envs to "base_conda_env", "named_conda_env_from_dict"],
        # and add local clusters once conda docker container is set up
    }
    MINIMAL = {
        "env": [
            "unnamed_env",
            "named_env",
            "base_conda_env",
            "named_conda_env_from_dict",
        ],
        "cluster": ["ondemand_aws_docker_cluster"],
    }
    RELEASE = {
        "env": [
            "unnamed_env",
            "named_env",
            "base_conda_env",
            "named_conda_env_from_dict",
        ],
        "cluster": [
            "ondemand_aws_docker_cluster",
            "static_cpu_pwd_cluster",
        ],
    }
    MAXIMAL = {
        "env": [
            "unnamed_env",
            "named_env",
            "base_conda_env",
            "named_conda_env_from_dict",
            "conda_env_from_local",
            "conda_env_from_path",
        ],
        "cluster": [
            "ondemand_aws_docker_cluster",
            "ondemand_gcp_cluster",
            "ondemand_k8s_cluster",
            "ondemand_k8s_docker_cluster",
            "ondemand_aws_https_cluster_with_auth",
            "static_cpu_pwd_cluster",
            "multinode_cpu_docker_conda_cluster",
            "docker_cluster_pk_ssh_no_auth",
            "docker_cluster_pwd_ssh_no_auth",
            "docker_cluster_pk_ssh_den_auth",
        ],
    }

    @pytest.mark.level("unit")
    def test_env_factory_and_properties(self, env):
        assert isinstance(env, rh.Env)

        args = init_args[id(env)]
        if "reqs" in args:
            assert set(args["reqs"]).issubset(env.reqs)

        if isinstance(env, rh.CondaEnv):
            assert env.conda_yaml
            assert isinstance(env.conda_yaml, Dict)
            assert set(["dependencies", "name"]).issubset(set(env.conda_yaml.keys()))

        if "name" not in args:
            assert not env.name

    @pytest.mark.level("unit")
    def test_env_conda_env_factories(self):
        name = "env_name"
        conda_env = {
            "name": "conda_env_name",
            "reqs": "pytest",
        }

        env = rh.env(name=name, conda_env=conda_env)
        conda_env = rh.conda_env(name=name, conda_env=conda_env)

        assert env.config() == conda_env.config()

    @pytest.mark.level("local")
    def test_env_to_cluster(self, env, cluster):
        env.to(cluster, force_install=True)

        for req in env.reqs:
            if not req == "./":
                res = cluster.run([f"pip freeze | grep {req}"], env=env)
                assert res[0][0] == 0  # installed properly

        _uninstall_env(env, cluster)

    @pytest.mark.skip("Running into s3 folder issue")
    @pytest.mark.level("minimal")
    def test_env_to_fs_to_cluster(self, env, cluster):
        s3_env = env.to("s3", force_install=True)
        for req in s3_env.reqs:
            if isinstance(req, rh.Package) and isinstance(
                req.install_target, rh.Folder
            ):
                assert req.install_target.system == "s3"
                assert req.install_target.exists_in_system()
                count += 1
        assert count >= 1

        folder_name = "test_package"
        count = 0
        conda_env_cluster = s3_env.to(
            system=cluster, path=folder_name, force_install=True
        )
        for req in conda_env_cluster.reqs:
            if isinstance(req, rh.Package) and isinstance(
                req.install_target, rh.Folder
            ):
                assert req.install_target.system == cluster
                count += 1
        assert count >= 1

        assert "sample_file_0.txt" in cluster.run([f"ls {folder_name}"])[0][1]
        cluster.run([f"rm -r {folder_name}"])

    @pytest.mark.level("local")
    def test_addtl_env_reqs(self, env, cluster):
        package = "jedi"
        env.reqs = env.reqs + [package] if env.reqs else [package]
        env.to(cluster, force_install=True)

        res = cluster.run([f"pip freeze | grep {package}"], env=env)
        assert res[0][0] == 0
        assert package in env.reqs

        _uninstall_env(env, cluster)

    @pytest.mark.level("local")
    def test_fn_to_env(self, env, cluster):
        package = "numpy"
        env.reqs = env.reqs + [package] or [package]
        fn = rh.function(np_summer).to(system=cluster, env=env, force_install=True)
        assert fn(1, 4) == 5

    @pytest.mark.level("local")
    def test_env_vars_dict(self, env, cluster):
        test_env_var = "TEST_ENV_VAR"
        test_value = "value"
        env.env_vars = {test_env_var: test_value}

        get_env_var_cpu = rh.function(_get_env_var_value).to(
            system=cluster, env=env, force_install=True
        )
        res = get_env_var_cpu(test_env_var)

        assert res == test_value

        _uninstall_env(env, cluster)

    @pytest.mark.level("local")
    def test_env_vars_file(self, env, cluster, tmp_path):
        env_file = str(tmp_path / ".env")
        contents = [
            "# comment",
            "",
            "ENV_VAR1=value",
            "# comment with =",
            "ENV_VAR2 =val2",
        ]
        with open(env_file, "w") as f:
            for line in contents:
                f.write(line + "\n")

        env.env_vars = env_file

        get_env_var_cpu = rh.function(_get_env_var_value).to(
            system=cluster, env=env, force_install=True
        )
        assert get_env_var_cpu("ENV_VAR1") == "value"
        assert get_env_var_cpu("ENV_VAR2") == "val2"

        os.remove(env_file)
        assert not Path(env_file).exists()

        _uninstall_env(env, cluster)

    @pytest.mark.level("local")
    def test_working_dir_env(self, env, cluster, tmp_path):
        dir_name = "test_working_dir"
        working_dir = tmp_path / dir_name
        working_dir.mkdir(exist_ok=True)
        env.working_dir = str(working_dir)

        assert str(working_dir) in env.reqs

        # Send the env to the cluster, save the dir in the main working directory (~) of the cluster
        env.to(cluster, path=dir_name, force_install=True)
        assert working_dir.name in cluster.run(["ls"])[0][1]

        cluster.run([f"rm -r {working_dir.name}"])
        assert working_dir.name not in cluster.run(["ls"])[0][1]

        _uninstall_env(env, cluster)

    @pytest.mark.level("local")
    def test_secrets_env(self, env, cluster):
        path_secret = rh.provider_secret(
            "lambda", values={"api_key": "test_api_key"}
        ).write(path="~/lambda_keys")
        api_key_secret = rh.provider_secret(
            "openai", values={"api_key": "test_openai_key"}
        )
        named_secret = (
            rh.provider_secret("huggingface", values={"token": "test_hf_token"})
            .write(path="~/hf_token")
            .save()
        )
        secrets = [path_secret, api_key_secret, named_secret.provider]

        env.secrets = secrets
        get_env_var_cpu = rh.function(_get_env_var_value).to(
            system=cluster, env=env, force_install=True
        )

        for secret in secrets:
            name = (
                secret if isinstance(secret, str) else (secret.name or secret.provider)
            )
            assert cluster.get(name)

            if isinstance(secret, str):
                secret = rh.Secret.from_name(secret)

            if secret.path:
                assert rh.folder(path=secret.path, system=cluster).exists_in_system()
            else:
                env_vars = secret.env_vars or secret._DEFAULT_ENV_VARS
                for _, var in env_vars.items():
                    assert get_env_var_cpu(var)

        named_secret.delete()

    @pytest.mark.level("local")
    def test_env_run_cmd(self, env, cluster):
        test_env_var = "ENV_VAR"
        test_value = "env_val"
        env.env_vars = {test_env_var: test_value}

        env.to(cluster)
        res = cluster.run(["echo $ENV_VAR"], env=env)

        assert res[0][0] == 0  # returncode
        assert "env_val" in res[0][1]  # stdout

    @pytest.mark.level("local")
    @pytest.mark.parametrize(
        "cmd",
        [
            "export ENV_VAR=env_val",
            "pip install numpy; echo $ENV_VAR",
            "pip freeze | grep numpy",
        ],
    )
    def test_env_run_shell_cmds(self, env, cluster, cmd):
        env.to(cluster)

        res = cluster.run([cmd], env=env)
        assert res[0][0] == 0

    @pytest.mark.level("local")
    def test_env_to_with_provider_secret(self, cluster):
        os.environ["HF_TOKEN"] = "test_hf_token"
        env = rh.env(name="hf_env", secrets=["huggingface"])
        env.to(cluster)

    @pytest.mark.level("local")
    def test_env_in_function_factory(self, cluster):
        remote_function = rh.function(scipy_import, env=["scipy"]).to(system=cluster)
        assert remote_function() is not None
