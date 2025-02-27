import importlib
import re
import time
from pathlib import Path
from threading import Thread

import pytest
import runhouse as rh
from runhouse.cli_utils import NodeFilterType
from runhouse.constants import DEFAULT_PROCESS_NAME, DEFAULT_SERVER_PORT
from runhouse.utils import capture_stdout

from tests.test_resources.test_clusters.test_cluster import sleep_fn
from tests.utils import get_pid_and_ray_node


class TestMultiNodeCluster:
    MAP_FIXTURES = {"resource": "cluster"}

    # setting release testing to run with on-demand cluster that not using docker image,
    # because the latter causing nightly release tests in CI to run for a very long time (does not happen locally).
    # TODO: [JL / SB]: check how we could make CI run with docker on-demand cluster

    UNIT = {"cluster": []}
    LOCAL = {"cluster": []}
    MINIMAL = {"cluster": []}
    RELEASE = {"cluster": ["multinode_k8s_cpu_cluster"]}
    MAXIMAL = {
        "cluster": [
            "multinode_k8s_cpu_cluster",
            "multinode_cpu_docker_conda_cluster",
            "multinode_gpu_cluster",
        ]
    }

    @pytest.mark.level("release")
    def test_rsync_and_ssh_onto_worker_node(self, cluster):
        worker_node = cluster.ips[-1]
        local_rh_package_path = Path(importlib.util.find_spec("runhouse").origin).parent

        local_rh_package_path = local_rh_package_path.parent
        dest_path = f"~/{local_rh_package_path.name}"

        # Rsync Runhouse package onto the worker node
        cluster.rsync(
            source=str(local_rh_package_path),
            dest=dest_path,
            up=True,
            node=worker_node,
            contents=True,
        )

        status_codes = cluster.run_bash([f"ls -l {dest_path}"], node=worker_node)
        assert status_codes[0][0] == 0

        assert "runhouse" in status_codes[0][1]

    @pytest.mark.level("release")
    def test_ray_started_on_worker_node_after_cluster_restart(self, cluster):
        head_node = cluster.head_ip

        status_codes = cluster.run_bash(["ray status"], node=head_node)
        assert status_codes[0][0] == 0

        status_output = status_codes[0][1]
        node_marker = "1 node_"
        num_nodes = status_output.count(node_marker)
        assert num_nodes == 2

    @pytest.mark.level("release")
    def test_send_envs_to_specific_worker_node(self, cluster):

        proc_0 = cluster.ensure_process_created("worker_0", compute={"node_idx": 0})
        proc_1 = cluster.ensure_process_created("worker_1", compute={"node_idx": 1})
        proc_2 = cluster.ensure_process_created("worker_2", compute={"node_idx": 1})

        with pytest.raises(ValueError):
            cluster.ensure_process_created(
                "worker_3", compute={"node_idx": len(cluster.ips)}
            )

        get_pid_0 = rh.function(get_pid_and_ray_node).to(
            name="get_pid_0", system=cluster, process=proc_0
        )
        get_pid_1 = rh.function(get_pid_and_ray_node).to(
            name="get_pid_1", system=cluster, process=proc_1
        )
        get_pid_2 = rh.function(get_pid_and_ray_node).to(
            name="get_pid_2", system=cluster, process=proc_2
        )

        with capture_stdout() as stdout_0:
            pid_0, node_id_0 = get_pid_0()
            assert str(pid_0) in str(stdout_0)
            assert str(node_id_0) in str(stdout_0)

        with capture_stdout() as stdout_1:
            pid_1, node_id_1 = get_pid_1()
            assert str(pid_1) in str(stdout_1)
            assert str(node_id_1) in str(stdout_1)

        with capture_stdout() as stdout_2:
            pid_2, node_id_2 = get_pid_2()
            assert str(pid_2) in str(stdout_2)
            assert str(node_id_2) in str(stdout_2)

        assert node_id_0 != node_id_1
        assert node_id_1 == node_id_2

    @pytest.mark.level("release")
    def test_specifying_resources(self, cluster):
        proc0 = cluster.ensure_process_created("worker_proc_0", compute={"CPU": 1.75})
        proc1 = cluster.ensure_process_created("worker_proc_1", compute={"CPU": 0.5})
        proc2 = cluster.ensure_process_created(
            "worker_proc_2", compute={"memory": 4 * 1024 * 1024 * 1024}
        )
        proc3 = cluster.ensure_process_created(
            "worker_proc_3", compute={"CPU": 0.1, "memory": 2 * 1024 * 1024 * 1024}
        )

        status = cluster.status()

        proc0_node = status["processes"][proc0]["node_ip"]
        proc1_node = status["processes"][proc1]["node_ip"]
        proc2_node = status["processes"][proc2]["node_ip"]
        proc3_node = status["processes"][proc3]["node_ip"]
        assert proc0_node in cluster.internal_ips
        assert proc1_node in cluster.internal_ips
        assert proc2_node in cluster.internal_ips
        assert proc3_node in cluster.internal_ips

        assert proc0_node != proc1_node  # Too much CPU
        assert proc2_node != proc3_node  # Too much memory

    @pytest.mark.level("release")
    def test_run_bash_on_node(self, cluster):
        # Specify via index
        result = cluster.run_bash("echo 'Hello World!'", node=0)
        assert result[0] == 0
        assert result[1] == "Hello World!\n"

        # Specify via IP
        result = cluster.run_bash("echo 'Hello World!'", node=cluster.ips[1])
        assert result[0] == 0
        assert result[1] == "Hello World!\n"

        # Run in process
        process = cluster.ensure_process_created("worker_env_0")
        result = cluster.run_bash("echo 'Hello World!'", process=process)
        assert result[0] == 0
        assert result[1] == "Hello World!\n"

    @pytest.mark.level("release")
    def test_multinode_secrets_to(self, cluster):
        custom_provider_secret = rh.provider_secret(
            provider="custom", values={"secret": "value"}
        )
        custom_provider_secret.to(cluster, path="~/.custom/secret.json")
        for node in cluster.ips:
            result = cluster.run_bash("ls ~/.custom", node=node)
            assert "secret.json" in result[1]

    @pytest.mark.level("release")
    def test_head_to_all_rsync(self, cluster):
        path = "/tmp/runhouse"
        cluster.run_bash(
            [f"mkdir -p {path}", f"echo 'hello there' > {path}/hello.txt"],
            node=cluster.head_ip,
            stream_logs=True,
        )

        cluster.rsync(
            source=path,
            dest=path,
            src_node=cluster.head_ip,
            node="all",
            contents=True,
            parallel=True,
        )

        for node in cluster.ips:
            status_codes = cluster.run_bash([f"ls -l {path}"], node=node)
            assert status_codes[0][0] == 0
            assert "hello.txt" in status_codes[0][1]
            status_codes = cluster.run_bash([f"cat {path}/hello.txt"], node=node)
            assert status_codes[0][0] == 0
            assert "hello there" in status_codes[0][1]

        # Test in the opposite direction, from worker to head, and parallel=False
        path = "/tmp/runhouse_2"
        cluster.run_bash(
            [f"mkdir -p {path}", f"echo 'hello again' > {path}/hello_again.txt"],
            node=cluster.ips[1],
            stream_logs=True,
        )

        cluster.rsync(
            source=path,
            dest=path,
            src_node=cluster.ips[1],
            node=cluster.head_ip,
            contents=True,
            parallel=False,
        )

        status_codes = cluster.run_bash([f"ls -l {path}"], node=cluster.head_ip)
        assert status_codes[0][0] == 0
        assert "hello_again.txt" in status_codes[0][1]
        status_codes = cluster.run_bash(
            [f"cat {path}/hello_again.txt"], node=cluster.head_ip
        )
        assert status_codes[0][0] == 0
        assert "hello again" in status_codes[0][1]

    ####################################################################################################
    # Status tests
    ####################################################################################################

    @pytest.mark.level("release")
    def test_rh_status_pythonic(self, cluster):
        process = cluster.ensure_process_created("worker")
        sleep_remote = rh.function(sleep_fn).to(cluster, process=process)
        cluster.put(key="status_key1", obj="status_value1", process=process)
        # Run these in a separate thread so that the main thread can continue
        call_threads = [Thread(target=sleep_remote, args=[3]) for _ in range(3)]
        for call_thread in call_threads:
            call_thread.start()

        # Wait a second so the calls can start
        time.sleep(1)
        cluster_data = cluster.status()

        # tests that the status output contains all expected keys
        expected_cluster_status_data_keys = [
            "processes",
            "server_pid",
            "runhouse_version",
            "cluster_config",
            "workers",
            "is_multinode",
        ]

        expected_worker_cpu_usage_keys = [
            "free_memory",
            "used_memory_percent",
            "total_memory",
            "used_memory",
            "utilization_percent",
        ]

        actual_cluster_status_data_keys = list(cluster_data.keys())

        for key in expected_cluster_status_data_keys:
            assert key in actual_cluster_status_data_keys

        # test that the status output contains information about all nodes.
        workers_info = cluster_data.get("workers")
        cluster_ips = cluster.ips
        assert len(workers_info) == len(cluster.ips)
        for worker_index in range(len(workers_info)):
            worker_resource_usage = workers_info[
                worker_index
            ]  # make sure that the workers match the ips list order
            assert worker_resource_usage.get("ip") == cluster_ips[worker_index]
            server_cpu_usage = worker_resource_usage.get("server_cpu_usage")
            assert isinstance(server_cpu_usage, dict)
            current_worker_cpu_usage_keys = server_cpu_usage.keys()
            for k in current_worker_cpu_usage_keys:
                assert k in expected_worker_cpu_usage_keys
            assert isinstance(server_cpu_usage.get("utilization_percent"), float)

        # test cluster config info
        res = cluster_data.get("cluster_config")
        assert res.get("server_port") == (cluster.server_port or DEFAULT_SERVER_PORT)
        assert res.get("server_connection_type") == cluster.server_connection_type
        assert res.get("den_auth") == cluster.den_auth
        assert res.get("resource_type") == cluster.RESOURCE_TYPE
        assert res.get("compute_properties").get("ips") == cluster.ips

        servlet_processes = cluster_data.get("processes")
        assert process in servlet_processes.keys()
        assert "status_key1" in servlet_processes.get(process).get(
            "process_resource_mapping"
        )
        assert {
            "resource_type": "str",
            "active_function_calls": [],
        } == servlet_processes.get(process).get("process_resource_mapping").get(
            "status_key1"
        )
        sleep_calls = (
            servlet_processes.get(process)
            .get("process_resource_mapping")
            .get("sleep_fn")
            .get("active_function_calls")
        )

        assert len(sleep_calls) == 3
        assert sleep_calls[0]["key"] == "sleep_fn"
        assert sleep_calls[0]["method_name"] == "call"
        assert sleep_calls[0]["request_id"]
        assert sleep_calls[0]["start_time"]

        # wait for threads to finish
        for call_thread in call_threads:
            call_thread.join()
        updated_status = cluster.status()
        updated_processes = updated_status.get("processes")
        # Check that the sleep calls are no longer active
        assert (
            updated_processes.get(process)
            .get("process_resource_mapping")
            .get("sleep_fn")
            .get("active_function_calls")
            == []
        )

        # test memory usage info
        expected_servlet_keys = [
            "node_index",
            "node_ip",
            "node_name",
            "pid",
            "process_cpu_usage",
            "process_resource_mapping",
        ]
        if cluster_data.get("cluster_config").get("is_gpu"):
            expected_servlet_keys.append("process_gpu_usage")
        expected_servlet_keys.sort()
        process_names = list(updated_processes.keys())
        process_names.sort()
        actors_keys = list(updated_processes.keys())
        actors_keys.sort()
        assert process_names == actors_keys
        for process_name in process_names:
            servlet_info = updated_processes.get(process_name)
            servlet_info_keys = list(servlet_info.keys())
            servlet_info_keys.sort()
            assert servlet_info_keys == expected_servlet_keys

    def status_cli_test_logic(
        self,
        cluster,
        status_cli_command: str,
        node: str = None,
        node_filter_type: NodeFilterType = None,
    ):
        default_process_name = DEFAULT_PROCESS_NAME

        for node_index in range(len(cluster.ips)):
            cluster.ensure_process_created(
                name=f"process_worker_{node_index}", compute={"node_idx": node_index}
            )
            cluster.put(
                key=f"status_key_worker_{node_index}",
                obj=f"status_value_worker_{node_index}",
                process=f"process_worker_{node_index}",
            )

        if node:
            status_cli_command = status_cli_command + f" --node {node}"

        status_output_response = cluster.run_bash([status_cli_command])[0]
        assert status_output_response[0] == 0
        status_output_string = status_output_response[1]
        # The string that's returned is utf-8 with the literal escape characters mixed in.
        # We need to convert the escape characters to their actual values to compare the strings.
        status_output_string = status_output_string.encode("utf-8").decode(
            "unicode_escape"
        )
        status_output_string = status_output_string.replace("\n", "")
        assert "Runhouse server is running" in status_output_string
        assert f"Runhouse v{rh.__version__}" in status_output_string
        assert f"server port: {cluster.server_port}" in status_output_string
        assert (
            f"server connection type: {cluster.server_connection_type}"
            in status_output_string
        )
        assert f"den auth: {str(cluster.den_auth)}" in status_output_string
        assert (
            f"resource subtype: {cluster.config().get('resource_subtype')}"
            in status_output_string
        )
        assert f"ips: {str(cluster.ips)}" in status_output_string
        assert "Serving " in status_output_string
        assert "creds" not in status_output_string

        # testing that CPU info of the server is printed correctly
        assert "CPU: " in status_output_string
        expected_workers_amount_in_output = 1 if node else len(cluster.ips)
        assert (
            status_output_string.count("CPU Utilization: ")
            == expected_workers_amount_in_output
        )
        assert (
            status_output_string.count("Serving") == expected_workers_amount_in_output
        )

        # testing that CPU info of the processes is printed correctly
        if not node:
            assert f"{default_process_name}" in status_output_string
            # checking default cli behaviour, info about all nodes should be printed
            for node_index in range(len(cluster.ips)):
                node_ip = cluster.ips[node_index]
                node_pattern = "head" if node_index == 0 else f"worker {node_index}"
                pattern = rf".*?{node_pattern} | IP: {node_ip} | CPU Utilization:.*?CPU:.*?status_key_worker_{node_index}"
                assert re.match(pattern, status_output_string)

        else:
            node_index = (
                int(node)
                if node_filter_type == NodeFilterType.node_index
                else cluster.ips.index(node)
            )
            node_name = "head node" if node_index == 0 else f"worker {node_index}"
            assert f"{node_name} | IP: {node}" in status_output_string
            assert f"status_key_worker_{node_index} (str)" in status_output_string

        cloud_properties = cluster.config().get("compute_properties")
        properties_to_check = ["cloud", "instance_type", "region", "cost_per_hour"]
        for p in properties_to_check:
            property_value = cloud_properties.get(p)
            assert property_value in status_output_string

    @pytest.mark.level("release")
    def test_rh_status_cmd_with_no_den_ping_default(self, cluster):
        self.status_cli_test_logic(
            cluster=cluster, status_cli_command="runhouse cluster status"
        )

    @pytest.mark.level("release")
    def test_rh_status_cmd_with_no_den_ping_with_node_ip(self, cluster):
        for ip in cluster.ips:
            self.status_cli_test_logic(
                cluster=cluster,
                status_cli_command="runhouse cluster status",
                node=ip,
                node_filter_type=NodeFilterType.ip,
            )

    @pytest.mark.level("release")
    def test_rh_status_cmd_with_no_den_ping_with_node_index(self, cluster):
        for node_index in range(len(cluster.ips)):
            self.status_cli_test_logic(
                cluster=cluster,
                status_cli_command="runhouse cluster status",
                node=node_index,
                node_filter_type=NodeFilterType.node_index,
            )

    @pytest.mark.level("release")
    def test_rh_status_cmd_with_den_ping_default(self, cluster):
        self.status_cli_test_logic(
            cluster=cluster, status_cli_command="runhouse cluster status --send-to-den"
        )

    @pytest.mark.level("release")
    def test_rh_status_cmd_with_den_ping_with_node_ip(self, cluster):
        for ip in cluster.ips:
            self.status_cli_test_logic(
                cluster=cluster,
                status_cli_command="runhouse cluster status --send-to-den",
                node=ip,
                node_filter_type=NodeFilterType.ip,
            )

    @pytest.mark.level("release")
    def test_rh_status_cmd_with_den_ping_with_node_index(self, cluster):
        for node_index in range(len(cluster.ips)):
            self.status_cli_test_logic(
                cluster=cluster,
                status_cli_command="runhouse cluster status --send-to-den",
                node=node_index,
                node_filter_type=NodeFilterType.node_index,
            )

    @pytest.mark.level("release")
    def test_send_status_to_db(self, cluster):

        from tests.test_resources.test_clusters.test_cluster import (
            send_cluster_status_to_db_logic,
        )

        send_cluster_status_to_db_logic(cluster)
