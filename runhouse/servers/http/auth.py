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
    def __init__(self):
        self.cache = {}

    def get_user_resources(self, token_hash: str) -> dict:
        """Get resources associated with a particular user's token"""
        return self.cache.get(token_hash, {})

    def lookup_access_level(
        self, token_hash: str, resource_uri: str
    ) -> Union[str, None]:
        resources: dict = self.get_user_resources(token_hash)
        return resources.get(resource_uri)

    def add_user(self, token, refresh_cache=True):
        """Refresh the server cache with the latest resources and access levels for a particular user"""
        if not refresh_cache and hash_token(token) in self.cache:
            return

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
        self.cache[hash_token(token)] = all_resources

    def clear_cache(self, token_hash: str = None):
        """Clear the server cache for a particular user's token"""
        if token_hash is None:
            self.cache = {}
        else:
            self.cache.pop(token_hash, None)


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
        obj_store.add_user(token)

        cached_resources: dict = obj_store.user_resources(token_hash)
        cluster_access_level = cached_resources.get(cluster_uri)

    if cluster_access_level in [ResourceAccess.WRITE, ResourceAccess.READ]:
        return True

    return False


def hash_token(token: str) -> str:
    """Hash the user's token to avoid storing them in plain text on the cluster."""
    return hashlib.sha256(token.encode()).hexdigest()
