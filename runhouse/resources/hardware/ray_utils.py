import subprocess
from typing import Optional

import ray
from ray.experimental.state.api import list_actors

from runhouse.logger import get_logger

logger = get_logger(__name__)


def check_for_existing_ray_instance():
    ray_status_check = subprocess.run(
        ["ray", "status"], stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    return ray_status_check.returncode == 0


def list_actor_states(
    actor_name: Optional[str] = None,
    actor_class_name: Optional[str] = None,
    namespace: Optional[str] = "runhouse",
    state: Optional[str] = "ALIVE",
):
    def filter_by(actor: "ActorState"):
        if actor_name and actor["name"] != actor_name:
            return False

        if actor_class_name and actor["class_name"] != actor_class_name:
            return False

        if namespace and actor["ray_namespace"] != namespace:
            return False

        if state and actor["state"] != state:
            return False

        return True

    return list(filter(filter_by, list_actors())) if ray.is_initialized() else []


def kill_actors(
    actor_name: Optional[str] = None,
    actor_class_name: Optional[str] = None,
    namespace: Optional[str] = None,
    gracefully: bool = True,
):
    cluster_servlet_actor = None
    for actor in list_actor_states(actor_name, actor_class_name, namespace):
        actor_handle_to_kill = ray.get_actor(actor["name"])
        if actor["name"] == "cluster_servlet":
            cluster_servlet_actor = actor_handle_to_kill
            continue
        logger.info(f"Killing actor {actor['name']}")
        if gracefully:
            actor_handle_to_kill.__ray_terminate__.remote()
        else:
            ray.kill(actor_handle_to_kill)

    # Make sure to kill cluster_servlet last
    if cluster_servlet_actor:
        logger.info("Killing actor cluster_servlet")
        if gracefully:
            cluster_servlet_actor.__ray_terminate__.remote()
        else:
            ray.kill(cluster_servlet_actor)
