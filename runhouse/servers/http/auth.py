import json
import logging
from typing import Union

import requests

from runhouse.globals import rns_client
from runhouse.rns.utils.api import load_resp_content, ResourceAccess
from runhouse.servers.http.http_utils import hash_token

logger = logging.getLogger(__name__)


class AuthCache:
    CACHE = {}

    @classmethod
    def get_user_resources(cls, token: str) -> dict:
        """Get resources associated with a particular user's token"""
        return cls.CACHE.get(hash_token(token), {})

    @classmethod
    def lookup_access_level(
        cls, token: str, resource_uri: str, retry=True
    ) -> Union[str, None]:
        resources: dict = cls.get_user_resources(token)
        if not resources and retry:
            cls.add_user(token)

            # Try again after refreshing the cache
            return cls.lookup_access_level(token, resource_uri, retry=False)

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
    to all other resources on the cluster. For calling functions must have read or write access to the cluster."""
    from runhouse.globals import obj_store

    # Check if user's token has already been validated and saved to cache on the cluster
    cached_resources: dict = obj_store.get_user_resources(token)

    if not cached_resources:
        # Refresh the cache with the resources user has access to
        obj_store.add_user(token)
        cached_resources: dict = obj_store.get_user_resources(token)

    # e.g. {"/jlewitt1/bert-preproc": "read"}
    cluster_access_type = cached_resources.get(cluster_uri)
    if cluster_access_type == ResourceAccess.WRITE.value:
        # If user has write access to the cluster will have access to all functions on the cluster
        return True

    # For running functions must have read or write access to the cluster
    if func_call and cluster_access_type not in [
        ResourceAccess.READ,
        ResourceAccess.WRITE,
    ]:
        return False

    return True
