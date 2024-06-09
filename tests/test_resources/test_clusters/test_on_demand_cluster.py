import json
import time

import pytest
import requests
import runhouse as rh
from runhouse.constants import (
    CLUSTER_CONFIG_PATH,
    DEFAULT_LOG_SURFACING_INTERVAL,
    DEFAULT_SURFACED_LOG_LENGTH,
    MAX_SURFACED_LOG_LENGTH,
    SERVER_LOGFILE_PATH,
)

from runhouse.resources.hardware.utils import remove_chars_from_str

import tests.test_resources.test_clusters.test_cluster
from tests.utils import friend_account


class TestOnDemandCluster(tests.test_resources.test_clusters.test_cluster.TestCluster):

    MAP_FIXTURES = {"resource": "cluster"}

    UNIT = {"cluster": []}
    LOCAL = {"cluster": []}
    MINIMAL = {
        "cluster": [
            "ondemand_aws_cluster",
            "ondemand_gcp_cluster",
        ]
    }
    RELEASE = {
        "cluster": [
            "ondemand_aws_cluster",
            "ondemand_gcp_cluster",
            # "ondemand_k8s_cluster",  # tested in test_kubernetes_cluster.py
            "ondemand_aws_https_cluster_with_auth",
        ]
    }
    MAXIMAL = {
        "cluster": [
            "ondemand_aws_cluster",
            "ondemand_gcp_cluster",
            "ondemand_k8s_cluster",
            "ondemand_aws_https_cluster_with_auth",
            "v100_gpu_cluster",
            "k80_gpu_cluster",
            "a10g_gpu_cluster",
            "static_cpu_cluster",
            "password_cluster",
            "multinode_cpu_cluster",
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
    def test_logs_surfacing_scheduler_basic_flow(self, cluster):
        if not cluster.den_auth:
            pytest.skip(
                "This test checking pinging cluster status to den, this could be done only on clusters "
                "with den_auth that can be saved to den."
            )

        # the scheduler start running in a delay of 2 min, so the cluster startup will finish properly.
        # Therefore, the test needs to sleep for a while.
        time.sleep(120)

        cluster_uri = rh.globals.rns_client.format_rns_address(cluster.rns_address)
        headers = rh.globals.rns_client.request_headers()
        api_server_url = rh.globals.rns_client.api_server_url

        get_status_data_resp = requests.get(
            f"{api_server_url}/resource/{cluster_uri}/logs",
            headers=headers,
        )

        # check that the cluster logs tail is updated correctly in the DB and in S3, and can be retrieved.
        chars_to_remove_from_logs = {
            "": "",
            "[2m[36m": "",
            " (EnvServlet": "(EnvServlet",
            " INFO": "INFO",
            "[0m": "",
        }
        cluster_logs = remove_chars_from_str(
            cluster.run([f"cat {SERVER_LOGFILE_PATH}"])[0][1], chars_to_remove_from_logs
        )

        assert "Trying to send cluster logs to Den" in cluster_logs

        assert get_status_data_resp.status_code == 200
        cluster_logs_from_s3 = get_status_data_resp.json()["data"][0]
        assert cluster_logs_from_s3 in cluster_logs

    @pytest.mark.level("minimal")
    def test_logs_surfacing_change_log_tail_length(self, cluster):
        if not cluster.den_auth:
            pytest.skip(
                "This test checking pinging cluster status to den, this could be done only on clusters "
                "with den_auth that can be saved to den."
            )

        # restarting the server in order to wait only 2 minutes for the logs scheduler to run.
        # (after restarting the server, first run of the scheduler is delayed by 2 minutes.)
        # TODO: check why restart_server() fails, probelm with putting default env for clusters with tls connection type.
        try:
            cluster.restart_server()
        except Exception as e:
            print(f"Restart failed. {e}")

        # Checking that when first setting up the scheduler, the default settings are set.
        cluster_config_before_change = json.loads(
            cluster.run([f"cat {CLUSTER_CONFIG_PATH}"])[0][1]
        )
        assert (
            cluster_config_before_change.get("surfaced_logs_length")
            == DEFAULT_SURFACED_LOG_LENGTH
        )
        assert (
            cluster_config_before_change.get("logs_surfacing_interval")
            == DEFAULT_LOG_SURFACING_INTERVAL
        )

        new_logs_length = 5
        cluster._enable_or_update_log_surface_to_den(num_of_lines=new_logs_length)
        # the scheduler start running in a delay of 2 min, so the cluster startup will finish properly.
        # Therefore, the test needs to sleep for a while.
        time.sleep(120)

        # Checking that the scheduler settings are saved correctly after the change.
        cluster_config_after_change = json.loads(
            cluster.run([f"cat {CLUSTER_CONFIG_PATH}"])[0][1]
        )
        assert (
            cluster_config_after_change.get("surfaced_logs_length") == new_logs_length
        )
        assert (
            cluster_config_before_change.get("logs_surfacing_interval")
            == DEFAULT_LOG_SURFACING_INTERVAL
        )

        cluster_uri = rh.globals.rns_client.format_rns_address(cluster.rns_address)
        headers = rh.globals.rns_client.request_headers()
        api_server_url = rh.globals.rns_client.api_server_url

        get_status_data_resp = requests.get(
            f"{api_server_url}/resource/{cluster_uri}/logs",
            headers=headers,
        )

        chars_to_remove_from_logs = {
            "": "",
            "[2m[36m": "",
            " (EnvServlet": "(EnvServlet",
            " INFO": "INFO",
            "[0m": "",
        }
        cluster_logs = remove_chars_from_str(
            cluster.run([f"cat {SERVER_LOGFILE_PATH}"])[0][1], chars_to_remove_from_logs
        )

        # checking the _enable_or_update_log_surface_to_den logic
        assert get_status_data_resp.status_code == 200
        cluster_logs_from_s3 = get_status_data_resp.json()["data"][0]
        assert cluster_logs_from_s3 in cluster_logs
        assert cluster_logs_from_s3.count("\n") == new_logs_length

        last_line_sent_to_s3 = cluster_logs_from_s3.split("\n")[:-1][-1]
        line_position_in_full_log_file = cluster_logs.split("\n").index(
            last_line_sent_to_s3
        )
        schedulers_logs = cluster_logs.split("\n")[
            line_position_in_full_log_file + 1 : line_position_in_full_log_file + 5
        ]
        schedulers_logs = [
            log
            for log in schedulers_logs
            if "Trying to send cluster logs to Den" in log
            or "Successfully sent cluster logs to Den." in log
        ]
        assert schedulers_logs
        assert len(schedulers_logs) == 2

    @pytest.mark.level("minimal")
    def test_logs_surfacing_change_log_tail_length_more_then_max(self, cluster, caplog):
        if not cluster.den_auth:
            pytest.skip(
                "This test checking pinging cluster status to den, this could be done only on clusters "
                "with den_auth that can be saved to den."
            )

        # restarting the server in order to wait only 2 minutes for the logs scheduler to run.
        # (after restarting the server, first run of the scheduler is delayed by 2 minutes.)
        try:
            cluster.restart_server()
        except Exception as e:
            print(f"Restart failed. {e}")

        # Checking that when first setting up the scheduler, the default settings are set.
        cluster_config_before_change = json.loads(
            cluster.run([f"cat {CLUSTER_CONFIG_PATH}"])[0][1]
        )
        assert (
            cluster_config_before_change.get("surfaced_logs_length")
            == DEFAULT_SURFACED_LOG_LENGTH
        )
        assert (
            cluster_config_before_change.get("logs_surfacing_interval")
            == DEFAULT_LOG_SURFACING_INTERVAL
        )

        new_logs_length = MAX_SURFACED_LOG_LENGTH + 5
        cluster._enable_or_update_log_surface_to_den(num_of_lines=new_logs_length)
        # the scheduler start running in a delay of 2 min, so the cluster startup will finish properly.
        # Therefore, the test needs to sleep for a while.
        time.sleep(120)

        # Checking that the scheduler settings are saved correctly after the change.
        cluster_config_after_change = json.loads(
            cluster.run([f"cat {CLUSTER_CONFIG_PATH}"])[0][1]
        )
        assert (
            cluster_config_after_change.get("surfaced_logs_length")
            == MAX_SURFACED_LOG_LENGTH
        )
        assert (
            cluster_config_before_change.get("logs_surfacing_interval")
            == DEFAULT_LOG_SURFACING_INTERVAL
        )

        cluster_uri = rh.globals.rns_client.format_rns_address(cluster.rns_address)
        headers = rh.globals.rns_client.request_headers()
        api_server_url = rh.globals.rns_client.api_server_url

        get_status_data_resp = requests.get(
            f"{api_server_url}/resource/{cluster_uri}/logs",
            headers=headers,
        )

        chars_to_remove_from_logs = {
            "": "",
            "[2m[36m": "",
            " (EnvServlet": "(EnvServlet",
            " INFO": "INFO",
            "[0m": "",
        }
        cluster_logs = remove_chars_from_str(
            cluster.run([f"cat {SERVER_LOGFILE_PATH}"])[0][1], chars_to_remove_from_logs
        )

        # checking the _enable_or_update_log_surface_to_den logic
        assert get_status_data_resp.status_code == 200
        cluster_logs_from_s3 = remove_chars_from_str(
            "".join(get_status_data_resp.json()["data"]), chars_to_remove_from_logs
        )
        assert cluster_logs_from_s3 in cluster_logs
        expected_warning_msg = f"Your pricing model doesn't all to set log length to {new_logs_length} lines. Setting to maximum length of {MAX_SURFACED_LOG_LENGTH} lines."
        assert expected_warning_msg in caplog.text
        assert any(
            record.levelname == "WARNING" and expected_warning_msg in record.message
            for record in caplog.records
        )

        cluster_config_on_cluster = json.loads(
            cluster.run([f"cat {CLUSTER_CONFIG_PATH}"])[0][1]
        )
        assert (
            cluster_config_on_cluster.get("surfaced_logs_length")
            == MAX_SURFACED_LOG_LENGTH
        )
        assert cluster_logs_from_s3.count("\n") == MAX_SURFACED_LOG_LENGTH

        last_line_sent_to_s3 = cluster_logs_from_s3.split("\n")[:-1][-1]
        line_position_in_full_log_file = cluster_logs.split("\n").index(
            last_line_sent_to_s3
        )
        schedulers_logs = cluster_logs.split("\n")[
            line_position_in_full_log_file + 1 : line_position_in_full_log_file + 5
        ]
        schedulers_logs = [
            log
            for log in schedulers_logs
            if "Trying to send cluster logs to Den" in log
            or "Successfully sent cluster logs to Den." in log
        ]
        assert schedulers_logs
        assert len(schedulers_logs) == 2

    @pytest.mark.level("minimal")
    def test_logs_surfacing_change_interval_size(self, cluster):
        if not cluster.den_auth:
            pytest.skip(
                "This test checking pinging cluster status to den, this could be done only on clusters "
                "with den_auth that can be saved to den."
            )

        # restarting the server in order to wait only 2 minutes for the logs scheduler to run.
        # (after restarting the server, first run of the scheduler is delayed by 2 minutes.)
        try:
            cluster.restart_server()
        except Exception as e:
            print(f"Restart failed. {e}")

        # Checking that when first setting up the scheduler, the default settings are set.
        cluster_config_before_change = json.loads(
            cluster.run([f"cat {CLUSTER_CONFIG_PATH}"])[0][1]
        )
        assert (
            cluster_config_before_change.get("surfaced_logs_length")
            == DEFAULT_SURFACED_LOG_LENGTH
        )
        assert (
            cluster_config_before_change.get("logs_surfacing_interval")
            == DEFAULT_LOG_SURFACING_INTERVAL
        )

        cluster._enable_or_update_log_surface_to_den(logs_surfacing_interval=60)
        # the scheduler start running in a delay of 2 min, so the cluster startup will finish properly.
        # Therefore, the test needs to sleep for a while.
        time.sleep(120)

        # Checking that the scheduler settings are saved correctly after the change.
        cluster_config_after_change = json.loads(
            cluster.run([f"cat {CLUSTER_CONFIG_PATH}"])[0][1]
        )
        assert (
            cluster_config_after_change.get("surfaced_logs_length")
            == DEFAULT_SURFACED_LOG_LENGTH
        )
        assert cluster_config_after_change.get("logs_surfacing_interval") == 60

        # checking the _enable_or_update_log_surface_to_den logic
        chars_to_remove_from_logs = {
            "": "",
            "[2m[36m": "",
            " (EnvServlet": "(EnvServlet",
            " INFO": "INFO",
            "[0m": "",
        }
        cluster_logs = remove_chars_from_str(
            cluster.run([f"cat {SERVER_LOGFILE_PATH}"])[0][1], chars_to_remove_from_logs
        )

        assert "Trying to send cluster logs to Den" in cluster_logs
        assert (
            "Successfully sent cluster logs to Den. Next status check will be in 1.0 minutes."
            in cluster_logs
        )

    @pytest.mark.level("minimal")
    def test_disable_logs_surfacing(self, cluster):
        if not cluster.den_auth:
            pytest.skip(
                "This test checking pinging cluster status to den, this could be done only on clusters "
                "with den_auth that can be saved to den."
            )

        # Restarting the server in order to wait only 2 minutes for the logs scheduler to run.
        # (after restarting the server, first run of the scheduler is delayed by 2 minutes.)
        try:
            cluster.restart_server()
        except Exception as e:
            print(f"Restart failed. {e}")

        # Checking that when first setting up the scheduler, the default settings are set.
        cluster_config_before_change = json.loads(
            cluster.run([f"cat {CLUSTER_CONFIG_PATH}"])[0][1]
        )
        assert (
            cluster_config_before_change.get("surfaced_logs_length")
            == DEFAULT_SURFACED_LOG_LENGTH
        )
        assert (
            cluster_config_before_change.get("logs_surfacing_interval")
            == DEFAULT_LOG_SURFACING_INTERVAL
        )

        cluster._disable_log_surface_to_den()
        # The scheduler start running in a delay of 2 min, so the cluster startup will finish properly.
        # Therefore, the test needs to sleep for a while.
        time.sleep(120)

        # Checking that the scheduler settings are saved correctly after the change.
        cluster_config_after_change = json.loads(
            cluster.run([f"cat {CLUSTER_CONFIG_PATH}"])[0][1]
        )
        assert cluster_config_after_change.get("surfaced_logs_length") == 0
        assert cluster_config_after_change.get("logs_surfacing_interval") == -1

        # checking the _disable_log_surface_to_den logic
        chars_to_remove_from_logs = {
            "": "",
            "[2m[36m": "",
            " (EnvServlet": "(EnvServlet",
            " INFO": "INFO",
            "[0m": "",
        }
        cluster_logs = remove_chars_from_str(
            cluster.run([f"cat {SERVER_LOGFILE_PATH}"])[0][1], chars_to_remove_from_logs
        )
        assert (
            "Disabled cluster logs surfacing. For enabling it, please run cluster.restart_server()."
        ) in cluster_logs
        assert (
            f"If you want to set the interval size and/or the log tail length to values that are not the default ones ({round(DEFAULT_LOG_SURFACING_INTERVAL / 60, 2)} minutes, {DEFAULT_SURFACED_LOG_LENGTH} lines), please run cluster._enable_or_update_log_surface_to_den(num_of_lines, interval_size) after restarting the server."
            in cluster_logs
        )
        assert "Trying to send cluster logs to Den" in cluster_logs
