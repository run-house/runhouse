import importlib
import logging
import shlex
import subprocess
from pathlib import Path

import pytest

from tests.fixtures.docker_cluster_fixtures import (
    popen_shell_command,
    run_shell_command,
)


@pytest.fixture(scope="session")
def local_telemetry_collector(request):
    """Local telemetry collector running in Docker which Mocks a Runhouse collector backend."""
    import docker

    force_rebuild = request.config.getoption("--force-rebuild")
    detached = request.config.getoption("--detached")

    rh_parent_path = Path(importlib.util.find_spec("runhouse").origin).parent.parent
    telemetry_docker_dir = rh_parent_path / "docker/telemetry-collector"
    config_file_path = telemetry_docker_dir / "otel-collector-config.yaml"

    image_name = "local-otel-collector"
    container_name = "otel"
    client = docker.from_env()

    containers = client.containers.list(
        all=True,
        filters={
            "ancestor": image_name,
            "name": container_name,
        },
    )
    if len(containers) > 0:
        if not detached:
            raise ValueError(
                f"Container {container_name} already running, but detached=False"
            )
        else:
            logging.info(
                f"Container {container_name} already running, skipping build and run."
            )
            return client

    images = client.images.list(filters={"reference": image_name})
    if not images or force_rebuild:
        build_cmd = [
            "docker",
            "build",
            "-t",
            image_name,
            "-f",
            str(telemetry_docker_dir / "Dockerfile"),
            str(telemetry_docker_dir),
        ]

        print(shlex.join(build_cmd))
        run_shell_command(subprocess, build_cmd, cwd=str(rh_parent_path.parent))

    # Note: using slightly different ports for http and grpc for this collector to avoid collisions if we are also
    # running the cluster Otel agent locally on the more standard ports (4318 and 4317)
    port_fwds = ["13134:13134", "4319:4319", "4316:4316"]
    port_fwds = (
        "".join([f"-p {port_fwd} " for port_fwd in port_fwds]).strip().split(" ")
    )
    # Run the Docker image
    run_cmd = (
        [
            "docker",
            "run",
            "--name",
            container_name,
            "-d",
            "--rm",
            "-v",
            f"{config_file_path}:/otel-collector-config.yaml",
        ]
        + port_fwds
        + [image_name]
    )
    print(shlex.join(run_cmd))
    res = popen_shell_command(subprocess, run_cmd, cwd=str(rh_parent_path.parent))
    stdout, stderr = res.communicate()
    if res.returncode != 0:
        raise RuntimeError(f"Failed to run otel docker image {image_name}: {stderr}")
