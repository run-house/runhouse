import importlib
from pathlib import Path

import pytest


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
