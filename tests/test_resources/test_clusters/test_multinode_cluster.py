import pkgutil
from pathlib import Path

import pytest


@pytest.mark.multinodetest
class TestMultiNodeCluster:
    @pytest.mark.level("thorough")
    def test_rsync_and_ssh_onto_worker_node(self, multinode_cpu_cluster):
        worker_node = multinode_cpu_cluster.ips[-1]
        local_rh_package_path = Path(pkgutil.get_loader("runhouse").path).parent

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

    @pytest.mark.level("thorough")
    def test_ray_started_on_worker_node_after_cluster_restart(
        self, multinode_cpu_cluster
    ):
        worker_node = multinode_cpu_cluster.ips[-1]

        multinode_cpu_cluster.restart_server(restart_ray=True)

        status_codes = multinode_cpu_cluster.run(["ray status"], node=worker_node)
        assert status_codes[0][0] == 0
