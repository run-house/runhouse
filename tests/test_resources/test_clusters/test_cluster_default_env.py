import pytest

import runhouse as rh

import tests.test_resources.test_clusters.test_cluster
from tests.test_resources.test_clusters.test_cluster import summer
from tests.test_resources.test_envs.test_env import _get_env_var_value

from tests.utils import get_random_str


def import_env():
    import pandas  # noqa
    import pytest  # noqa

    return "success"


class TestCluster(tests.test_resources.test_clusters.test_cluster.TestCluster):
    MAP_FIXTURES = {"resource": "cluster"}

    UNIT = {"cluster": []}
    LOCAL = {"cluster": ["docker_cluster_pk_ssh"]}
    MINIMAL = {"cluster": ["ondemand_default_conda_env_cluster"]}
    RELEASE = {"cluster": ["ondemand_default_conda_env_cluster"]}
    MAXIMAL = {"cluster": ["ondemand_default_conda_env_cluster"]}

    @pytest.mark.level("local")
    def test_default_env_in_status(self, cluster):
        res = cluster.status()
        assert cluster.default_env.name in res.get("envs")

    @pytest.mark.level("local")
    def test_put_in_default_env(self, cluster):
        k1 = get_random_str()
        cluster.put(k1, "v1")

        assert k1 in cluster.keys(env=cluster.default_env.name)
        cluster.delete(k1)

    @pytest.mark.level("local")
    def test_fn_to_default_env(self, cluster):
        remote_summer = rh.function(summer).to(cluster)

        assert remote_summer.name in cluster.keys(env=cluster.default_env.name)
        assert remote_summer(3, 4) == 7

    @pytest.mark.level("local")
    def test_run_in_default_env(self, cluster):
        for req in cluster.default_env.reqs:
            if isinstance(req, str) and "_" in req:
                # e.g. pytest_asyncio
                req = req.replace("_", "-")
                assert cluster.run(f"pip freeze | grep {req}")[0][0] == 0

    @pytest.mark.level("minimal")
    def test_default_conda_env_created(self, cluster):
        conda_env = cluster.default_env

        assert cluster.default_env.env_name in cluster.run("conda info --envs")[0][1]
        assert isinstance(cluster.get(conda_env.name), rh.CondaEnv)

    @pytest.mark.level("release")
    def test_switch_default_env(self, cluster):
        # test setting a new default env, w/o restarting the runhouse server
        test_env = cluster.default_env
        new_env = rh.conda_env(name="new_conda_env", reqs=["diffusers"])
        cluster.default_env = new_env

        # check cluster attr set, and  new env exists on the system
        assert new_env.env_name in cluster.run("conda info --envs")[0][1]
        assert cluster.default_env.name == new_env.name
        assert new_env.name in cluster.status().get("envs")

        # check that env defaults to new default env for run/put
        assert cluster.run("pip freeze | grep diffusers")[0][0] == 0

        k1 = get_random_str()
        cluster.put(k1, "v1")
        assert k1 in cluster.keys(env=new_env.env_name)

        # set it back
        cluster.default_env = test_env
        cluster.delete(new_env.name)

    @pytest.mark.level("release")
    def test_default_env_var_run(self, cluster):
        env_vars = cluster.default_env.env_vars

        assert env_vars
        for var in env_vars.keys():
            res = cluster.run([f"echo ${var}"], env=cluster.default_env)
            assert res[0][0] == 0
            assert env_vars[var] in res[0][1]

    @pytest.mark.level("release")
    def test_default_env_var_fn(self, cluster):
        env_vars = cluster.default_env.env_vars

        get_env_var_cpu = rh.function(_get_env_var_value).to(system=cluster)
        for var in env_vars.keys():
            assert get_env_var_cpu(var) == env_vars[var]

    @pytest.mark.level("local")
    def test_fn_env_var_import(self, cluster):
        fn = rh.function(import_env).to(cluster)
        assert fn() == "success"
