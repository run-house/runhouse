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
            raise Exception(
                f"Failed to load resources for user: {load_resp_content(resp)}"
            )

        resp_data = json.loads(resp.content)
        all_resources: dict = {
            resource["name"]: resource["access_type"] for resource in resp_data["data"]
        }

        # Update server cache with a user's resources and access type
        cls.CACHE[hash_token(token)] = all_resources
        logger.info(f"Updated cache for user with {len(all_resources)} resources:")


def verify_cluster_access(
    cluster_uri: str,
    token: str,
    func_call: bool,
) -> bool:
    """Verify the user has access to the cluster. If user has write access to the cluster, will have access
    to all other resources on the cluster. For calling functions must have read or write access to the cluster.
    Access is managed by the object store, with a Ray actor (AuthCache) which maps a user's hashed token to
    the list of resources they have access to."""
    from runhouse.globals import obj_store

    token_hash = hash_token(token)

    # Check if user's token has already been validated and saved to cache on the cluster
    cached_resources: dict = obj_store.user_resources(token_hash)

    # e.g. {"/jlewitt1/bert-preproc": "read"}
    cluster_access_type = cached_resources.get(cluster_uri)
    if cluster_access_type == ResourceAccess.WRITE.value:
        # If user has write access to the cluster will have access to all functions on the cluster
        return True

    # Refresh cache for the user
    # Note: for adding user we need to pass in the token since it requires querying Den
    auth_cache_actor = ray.get_actor("auth_cache", namespace="runhouse")
    ray.get(auth_cache_actor.add_user.remote(token))

    # Re-check if user has access to the cluster
    cached_resources: dict = obj_store.user_resources(token_hash)
    cluster_access_type = cached_resources.get(cluster_uri)

    # For running functions must have read or write access to the cluster
    if func_call and cluster_access_type not in [
        ResourceAccess.READ,
        ResourceAccess.WRITE,
    ]:
        return False

    return True


def hash_token(token: str) -> str:
    """Hash the user's token to avoid storing them in plain text on the cluster."""
    return hashlib.sha256(token.encode()).hexdigest()
