import importlib
from pathlib import Path

import pytest
import runhouse as rh

from tests.utils import get_pid_and_ray_node


class TestMultiNodeCluster:

    MAP_FIXTURES = {"resource": "cluster"}

    UNIT = {"cluster": []}
    LOCAL = {"cluster": []}
    MINIMAL = {"cluster": []}
    RELEASE = {"cluster": ["multinode_cpu_docker_conda_cluster"]}
    MAXIMAL = {
        "cluster": [
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

        status_codes = cluster.run([f"ls -l {dest_path}"], node=worker_node)
        assert status_codes[0][0] == 0

        assert "runhouse" in status_codes[0][1]

    @pytest.mark.level("release")
    def test_ray_started_on_worker_node_after_cluster_restart(self, cluster):
        head_node = cluster.ips[0]

        status_codes = cluster.run(["ray status"], node=head_node)
        assert status_codes[0][0] == 0

        status_output = status_codes[0][1]
        node_marker = "1 node_"
        num_nodes = status_output.count(node_marker)
        assert num_nodes == 2

    @pytest.mark.level("release")
    def test_send_envs_to_specific_worker_node(self, cluster):

        env_0 = rh.env(
            name="worker_env_0",
            reqs=["langchain", "pytest"],
        ).to(cluster, node_idx=0)

        env_1 = rh.env(
            name="worker_env_1",
            reqs=["torch", "pytest"],
        ).to(cluster, node_idx=1)

        env_2 = rh.env(
            name="worker_env_2",
            reqs=["transformers", "pytest"],
        )

        with pytest.raises(ValueError):
            env_2.to(
                cluster,
                node_idx=len(cluster.ips),
            )

        env_2.to(cluster, node_idx=1)

        get_pid_0 = rh.function(get_pid_and_ray_node).to(
            name="get_pid_0", system=cluster, env=env_0
        )
        get_pid_1 = rh.function(get_pid_and_ray_node).to(
            name="get_pid_1", system=cluster, env=env_1
        )
        get_pid_2 = rh.function(get_pid_and_ray_node).to(
            name="get_pid_2", system=cluster, env=env_2
        )
        assert get_pid_0()[1] != get_pid_1()[1]
        assert get_pid_1()[1] == get_pid_2()[1]

    @pytest.mark.level("release")
    def test_specifying_resources(self, cluster):
        env0 = rh.env(
            name="worker_env_0",
            compute={"CPU": 1.75},
        ).to(cluster)

        env1 = rh.env(
            name="worker_env_1",
            compute={"CPU": 0.5},
        ).to(cluster)

        env2 = rh.env(
            name="worker_env_2",
            compute={"memory": 4 * 1024 * 1024 * 1024},
        ).to(cluster)

        env3 = rh.env(
            name="worker_env_3",
            compute={"CPU": 0.1, "memory": 2 * 1024 * 1024 * 1024},
        ).to(cluster)

        status = cluster.status()

        env0_node = status["env_servlet_processes"][env0.name]["node_ip"]
        env1_node = status["env_servlet_processes"][env1.name]["node_ip"]
        env2_node = status["env_servlet_processes"][env2.name]["node_ip"]
        env3_node = status["env_servlet_processes"][env3.name]["node_ip"]
        assert env0_node in cluster.internal_ips
        assert env1_node in cluster.internal_ips
        assert env2_node in cluster.internal_ips
        assert env3_node in cluster.internal_ips

        assert env0_node != env1_node  # Too much CPU
        assert env2_node != env3_node  # Too much memory
