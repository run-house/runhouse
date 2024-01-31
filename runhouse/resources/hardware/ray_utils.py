import logging
from typing import Optional

import ray
from ray.experimental.state.api import list_actors

logger = logging.getLogger(__name__)


def check_for_existing_ray_instance(address: str, remain_connected: bool = False):
    try:
        ray.init(
            address=address,
            ignore_reinit_error=True,
            logging_level=logging.ERROR,
        )

        # Note that this is _technically_ just to check if there is an existing
        # cluster. We want to shutdown() which doesn't actually kill the cluster,
        # but cleans up the state of Ray within this Python process.
        if not remain_connected:
            ray.shutdown()
        return True
    except ConnectionError:
        return False


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
    for actor in list_actor_states(actor_name, actor_class_name, namespace):
        logger.info(f"Killing actor {actor['name']}")
        actor_handle_to_kill = ray.get_actor(actor["name"])
        if gracefully:
            actor_handle_to_kill.__ray_terminate__.remote()
        else:
            ray.kill(actor_handle_to_kill)
