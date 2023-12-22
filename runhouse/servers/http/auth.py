import hashlib
import json
import logging
from typing import Union

import ray
import requests

from runhouse.globals import rns_client
from runhouse.rns.utils.api import load_resp_content, ResourceAccess

logger = logging.getLogger(__name__)


class AuthCache:
    # Maps a user's token to all the resources they have access to
    CACHE = {}

    @classmethod
    def get_user_resources(cls, token_hash: str) -> dict:
        """Get resources associated with a particular user's token"""
        return cls.CACHE.get(token_hash, {})

    @classmethod
    def lookup_access_level(
        cls, token_hash: str, resource_uri: str
    ) -> Union[str, None]:
        resources: dict = cls.get_user_resources(token_hash)
        return resources.get(resource_uri)

    @classmethod
    def add_user(cls, token):
        """Refresh the server cache with the latest resources and access levels for a particular user"""
        resp = requests.get(
            f"{rns_client.api_server_url}/resource",
            headers={"Authorization": f"Bearer {token}"},
        )
        if resp.status_code != 200:
            logger.error(
                f"Failed to load resources for user: {load_resp_content(resp)}"
            )
            return

        resp_data = json.loads(resp.content)
        # Support access_level and access_type for BC
        all_resources: dict = {
            resource["name"]: resource.get("access_level")
            or resource.get("access_type")
            for resource in resp_data["data"]
        }
        # Update server cache with a user's resources and access type
        cls.CACHE[hash_token(token)] = all_resources


def verify_cluster_access(
    cluster_uri: str,
    token: str,
) -> bool:
    """Checks whether the user has access to the cluster.
    Note: If user has write access to the cluster, will have access to all other resources on the cluster by default."""
    from runhouse.globals import obj_store

    token_hash = hash_token(token)

    # Check if user already has saved resources in cache
    cached_resources: dict = obj_store.user_resources(token_hash)

    # e.g. {"/jlewitt1/bert-preproc": "read"}
    cluster_access_level = cached_resources.get(cluster_uri)

    if cluster_access_level is None:
        # Reload from cache and check again
        update_cache_for_user(token)

        cached_resources: dict = obj_store.user_resources(token_hash)
        cluster_access_level = cached_resources.get(cluster_uri)

    if cluster_access_level in [ResourceAccess.WRITE, ResourceAccess.READ]:
        return True

    return False


def hash_token(token: str) -> str:
    """Hash the user's token to avoid storing them in plain text on the cluster."""
    return hashlib.sha256(token.encode()).hexdigest()


def update_cache_for_user(token):
    auth_cache_actor = ray.get_actor("auth_cache", namespace="runhouse")
    ray.get(auth_cache_actor.add_user.remote(token))
