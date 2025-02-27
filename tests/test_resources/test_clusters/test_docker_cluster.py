import importlib

import os
from pathlib import Path

import pytest
import ray
import runhouse as rh


def get_uname():
    return os.uname()


@pytest.mark.level("release")
def test_docker_cluster():
    import docker

    client = docker.from_env()

    cluster = rh.DockerCluster(
        name="test-cluster",
        container_name="runhouse-test-container",
    )
    if not cluster.is_up():
        rh_parent_path = Path(importlib.util.find_spec("runhouse").origin).parent.parent
        dockerfile_path = rh_parent_path / "docker/slim"
        # Rebuild the image if not already built
        if not client.images.list(name="runhouse-slim"):
            client.images.build(
                path=".",
                dockerfile=str(dockerfile_path),
                tag="runhouse-slim",
            )
        container = client.containers.run(
            "runhouse-slim",
            command="tail -f /dev/null",
            detach=True,
            ports={"32300": 32300},
            shm_size="3gb",  # Needed by Ray
            name="runhouse-test-container",
        )
        container.start()
    # Installs the local runhouse version inside the container and starts the server,
    # skip if you've pre-installed runhouse[server] in the image and started the server in the docker CMD
    cluster.restart_server()

    cluster.install_packages(["pytest"])

    ray_resources = rh.function(ray.available_resources).to(cluster, sync_local=False)
    assert ray_resources()

    get_uname_dc = rh.function(get_uname).to(cluster)
    assert get_uname_dc()
