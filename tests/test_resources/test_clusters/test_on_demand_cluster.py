import asyncio
import time

import pytest
import requests

import runhouse as rh

from runhouse.globals import rns_client
from runhouse.resources.hardware.utils import ResourceServerStatus

import tests.test_resources.test_clusters.test_cluster
from tests.utils import friend_account


def set_autostop_from_on_cluster_via_ah(mins):
    ah = rh.servers.autostop_helper.AutostopHelper()

    asyncio.run(ah.set_autostop(mins))


def get_auotstop_from_on_cluster():
    ah = rh.servers.autostop_helper.AutostopHelper()

    return asyncio.run(ah.get_autostop())


def get_last_active_time_from_on_cluster():
    ah = rh.servers.autostop_helper.AutostopHelper()

    return asyncio.run(ah.get_last_active_time())


def register_activity_from_on_cluster():
    ah = rh.servers.autostop_helper.AutostopHelper()

    asyncio.run(ah.set_last_active_time_to_now())
    asyncio.run(ah.register_activity_if_needed())


def set_autostop_from_on_cluster_via_cluster_obj(mins):
    rh.here.autostop_mins = mins


def set_autostop_from_on_cluster_via_cluster_keep_warm():
    rh.here.keep_warm()


def torch_exists():
    try:
        import torch

        torch.rand(4)
        return True
    except ImportError:
        return False


class TestOnDemandCluster(tests.test_resources.test_clusters.test_cluster.TestCluster):

    MAP_FIXTURES = {"resource": "cluster"}

    UNIT = {"cluster": []}
    LOCAL = {"cluster": []}
    MINIMAL = {
        "cluster": [
            "ondemand_aws_docker_cluster",
            "ondemand_gcp_cluster",
            "ondemand_k8s_cluster",
        ]
    }
    RELEASE = {
        "cluster": [
            "ondemand_aws_docker_cluster",
            "ondemand_gcp_cluster",
            "ondemand_aws_https_cluster_with_auth",
            "ondemand_k8s_cluster",
            "ondemand_k8s_docker_cluster",
        ]
    }
    MAXIMAL = {
        "cluster": [
            "ondemand_aws_docker_cluster",
            "ondemand_gcp_cluster",
            "ondemand_k8s_cluster",
            "ondemand_k8s_docker_cluster",
            "ondemand_aws_https_cluster_with_auth",
            "v100_gpu_cluster",
            "k80_gpu_cluster",
            "a10g_gpu_cluster",
            "static_cpu_pwd_cluster",
            "multinode_cpu_docker_conda_cluster",
            "multinode_gpu_cluster",
        ]
    }

    @pytest.mark.level("minimal")
    def test_restart_does_not_change_config_yaml(self, cluster):
        assert cluster.up_if_not()
        config_yaml_res = cluster.run("cat ~/.rh/config.yaml")
        assert config_yaml_res[0][0] == 0
        config_yaml_content = config_yaml_res[0][1]

        cluster.share(
            users=["info@run.house"],
            access_level="read",
            notify_users=False,
        )

        with friend_account():
            cluster.restart_server()
            config_yaml_res_after_restart = cluster.run("cat ~/.rh/config.yaml")
            assert config_yaml_res_after_restart[0][0] == 0
            config_yaml_content_after_restart = config_yaml_res[0][1]
            assert config_yaml_content_after_restart == config_yaml_content

    @pytest.mark.level("minimal")
    def test_set_autostop(self, cluster):
        rh.env(
            working_dir="local:./", reqs=["pytest", "pandas"], name="autostop_env"
        ).to(cluster)

        get_autostop = rh.fn(get_auotstop_from_on_cluster).to(
            cluster, env="autostop_env"
        )
        # First check that the autostop is set to whatever the cluster set it to
        assert get_autostop() == cluster.autostop_mins
        original_autostop = cluster.autostop_mins

        set_autostop = rh.fn(set_autostop_from_on_cluster_via_ah).to(
            cluster, env="autostop_env"
        )
        set_autostop(5)
        assert get_autostop() == 5

        set_autostop_via_cluster_keep_warm = rh.fn(
            set_autostop_from_on_cluster_via_cluster_keep_warm
        ).to(cluster, env="autostop_env")
        set_autostop_via_cluster_keep_warm()
        assert get_autostop() == -1

        set_autostop_via_cluster_obj = rh.fn(
            set_autostop_from_on_cluster_via_cluster_obj
        ).to(cluster, env="autostop_env")
        # reset the autostop to the original value
        set_autostop_via_cluster_obj(original_autostop)
        assert get_autostop() == original_autostop

    @pytest.mark.level("minimal")
    def test_register_activity(self, cluster):
        rh.env(
            working_dir="local:./", reqs=["pytest", "pandas"], name="autostop_env"
        ).to(cluster)

        register_activity = rh.fn(register_activity_from_on_cluster).to(
            cluster, env="autostop_env"
        )
        get_last_active = rh.fn(get_last_active_time_from_on_cluster).to(
            cluster, env="autostop_env"
        )

        register_activity()
        # Check that last active is within the last 2 seconds
        assert get_last_active() > time.time() - 3

    # TODO add a way to manually trigger the status loop to check that activity
    #  is actually registered after a call
    # cluster.call("autostop_env", "config")
    # cluster.status()
    # assert get_last_active() > time.time() - 2

    # TODO add a way to manually trigger the status loop to check that activity
    #  is actually registered during a long running function
    # from .test_cluster import sleep_fn
    # sleep_remote = rh.fn(sleep_fn).to(cluster, env="autostop_env")
    # Thread(target=sleep_remote, args=(3,)).start()
    # time.sleep(2)
    # cluster.status()
    # # Check that last active is within the last second, so we know the activity wasn't just from the call itself
    # assert get_last_active() > time.time() - 1

    @pytest.mark.level("release")
    def test_cluster_ping_and_is_up(self, cluster):
        assert cluster._ping(retry=False)

        original_ips = cluster.ips

        cluster.address = None
        assert not cluster._ping(retry=False)

        if not (
            hasattr(cluster, "launched_properties")
            and cluster.launched_properties["cloud"] == "kubernetes"
        ):  # kubernetes does not use ips in command runner
            cluster.address = "00.00.000.11"
            assert not cluster._ping(retry=False)

        assert cluster._ping(retry=True)
        assert cluster.is_up()
        assert cluster.ips == original_ips

    @pytest.mark.level("release")
    def test_docker_container_reqs(self, ondemand_aws_docker_cluster):
        ret_code = ondemand_aws_docker_cluster.run("pip freeze | grep torch")[0][0]
        assert ret_code == 0

    @pytest.mark.level("release")
    def test_fn_to_docker_container(self, ondemand_aws_docker_cluster):
        remote_torch_exists = rh.function(torch_exists).to(ondemand_aws_docker_cluster)
        assert remote_torch_exists()

    ####################################################################################################
    # Status tests
    ####################################################################################################

    # TODO: Affects cluster state, causes other tests to fail with ssh connection errors
    @pytest.mark.skip()
    @pytest.mark.level("minimal")
    def test_set_status_after_teardown(self, cluster, mocker):
        mock_function = mocker.patch("sky.down")
        response = cluster.teardown()
        assert isinstance(response, int)
        assert (
            response == 200
        )  # that means that the call to post status endpoint in den was successful
        mock_function.assert_called_once()

        cluster_config = cluster.config()
        cluster_uri = rns_client.format_rns_address(cluster.rns_address)
        api_server_url = cluster_config.get("api_server_url", rns_client.api_server_url)
        cluster.teardown()
        get_status_data_resp = requests.get(
            f"{api_server_url}/resource/{cluster_uri}/cluster/status",
            headers=rns_client.request_headers(),
        )

        assert get_status_data_resp.status_code == 200
        # For UI displaying purposes, the cluster/status endpoint returns cluster status history.
        # The latest status info is the first element in the list returned by the endpoint.
        get_status_data = get_status_data_resp.json()["data"][0]
        assert get_status_data["resource_type"] == cluster_config.get("resource_type")
        assert get_status_data["status"] == ResourceServerStatus.terminated

        assert cluster.is_up()
