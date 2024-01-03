import pkgutil
from pathlib import Path

import pytest

import tests.test_resources.test_resource


class TestMultiNodeCluster(tests.test_resources.test_resource.TestResource):
    MAP_FIXTURES = {"resource": "cluster"}

    UNIT = {"cluster": []}
    LOCAL = {"cluster": []}
    MINIMAL = {"cluster": []}
    THOROUGH = {"cluster": ["multinode_cpu_cluster"]}
    MAXIMAL = {"cluster": ["multinode_cpu_cluster"]}

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
