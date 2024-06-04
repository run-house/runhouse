import importlib
from pathlib import Path

import pytest
import runhouse as rh

from tests.utils import get_pid_and_ray_node


class TestMultiNodeCluster:
    @pytest.mark.level("release")
    def test_rsync_and_ssh_onto_worker_node(self, multinode_cpu_cluster):
        worker_node = multinode_cpu_cluster.ips[-1]
        local_rh_package_path = Path(importlib.util.find_spec("runhouse").origin).parent

        local_rh_package_path = local_rh_package_path.parent
        dest_path = f"~/{local_rh_package_path.name}"

        # Rsync Runhouse package onto the worker node
        multinode_cpu_cluster._rsync(
            source=str(local_rh_package_path),
            dest=dest_path,
            up=True,
            node=worker_node,
            contents=True,
        )

        status_codes = multinode_cpu_cluster.run(["ls -l", dest_path], node=worker_node)
        assert status_codes[0][0] == 0

        assert "runhouse" in status_codes[0][1]

    @pytest.mark.level("release")
    def test_ray_started_on_worker_node_after_cluster_restart(
        self, multinode_cpu_cluster
    ):
        head_node = multinode_cpu_cluster.ips[0]

        status_codes = multinode_cpu_cluster.run(["ray status"], node=head_node)
        assert status_codes[0][0] == 0

        status_code_strings = []

        for status_code in status_codes:
            # Convert each element of the tuple to a string and join them with ", "
            status_code_string = ", ".join(map(str, status_code))
            status_code_strings.append(status_code_string)

        return_value = ", ".join(status_code_strings)
        node_marker = "1 node_"
        num_nodes = return_value.count(node_marker)

        assert num_nodes == 2

    @pytest.mark.level("release")
    def test_send_envs_to_specific_worker_node(self, multinode_cpu_cluster):

        env_0 = rh.env(
            name="worker_env_0",
            reqs=["langchain", "pytest"],
        ).to(multinode_cpu_cluster, node_idx=0)

        env_1 = rh.env(
            name="worker_env_1",
            reqs=["torch", "pytest"],
        ).to(multinode_cpu_cluster, node_idx=1)

        env_2 = rh.env(
            name="worker_env_2",
            reqs=["transformers", "pytest"],
        )

        with pytest.raises(ValueError):
            env_2.to(multinode_cpu_cluster, node_idx=len(multinode_cpu_cluster.ips))

        env_2.to(multinode_cpu_cluster, node_idx=1)

        get_pid_0 = rh.function(get_pid_and_ray_node).to(
            name="get_pid_0", system=multinode_cpu_cluster, env=env_0
        )
        get_pid_1 = rh.function(get_pid_and_ray_node).to(
            name="get_pid_1", system=multinode_cpu_cluster, env=env_1
        )
        get_pid_2 = rh.function(get_pid_and_ray_node).to(
            name="get_pid_2", system=multinode_cpu_cluster, env=env_2
        )
        assert get_pid_0()[1] != get_pid_1()[1]
        assert get_pid_1()[1] == get_pid_2()[1]
