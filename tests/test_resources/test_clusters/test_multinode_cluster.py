import importlib
from pathlib import Path

import pytest
import runhouse as rh
from runhouse.utils import capture_stdout

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

        proc0_node = status["env_servlet_processes"][proc0]["node_ip"]
        proc1_node = status["env_servlet_processes"][proc1]["node_ip"]
        proc2_node = status["env_servlet_processes"][proc2]["node_ip"]
        proc3_node = status["env_servlet_processes"][proc3]["node_ip"]
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
