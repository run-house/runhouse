import json
import logging
from typing import Union

from runhouse.globals import rns_client
from runhouse.rns.utils.api import load_resp_content, ResourceAccess

logger = logging.getLogger(__name__)


class AuthCache:
    # Maps a user's token to all the resources they have access to
    CACHE = {}

    @classmethod
    def get_user_resources(cls, username: str) -> dict:
        """Get resources associated with a particular username"""
        return cls.CACHE.get(username, {})

    @classmethod
    def lookup_access_level(cls, username: str, resource_uri: str) -> Union[str, None]:
        resources: dict = cls.get_user_resources(username)
        return resources.get(resource_uri)

    @classmethod
    def add_user(cls, username, token, refresh_cache=True):
        """Refresh the server cache with the latest resources and access levels for a particular username"""
        if username is None:
            return

        if not refresh_cache and username in cls.CACHE:
            return

        resp = rns_client.session.get(
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
        cls.CACHE[username] = all_resources

    def clear_cache(self, username: str = None):
        """Clear the server cache, If a username is specified, clear the cache for that particular user only"""
        if username is None:
            self.CACHE = {}
        else:
            self.CACHE.pop(username, None)


def verify_cluster_access(
    cluster_uri: str,
    username: str,
    token: str,
) -> bool:
    """Checks whether the user has access to the cluster.
    Note: A user with write access to the cluster or a cluster owner will have access to all other resources on
    the cluster by default."""
    from runhouse.globals import configs, obj_store

    # The logged-in user always has full access to the cluster. This is especially important if they flip on
    # Den Auth without saving the cluster. We may need to generate a subtoken here to check.
    if configs.token and (
        configs.token == token
        or rns_client.cluster_token(configs.token, cluster_uri) == token
    ):
        return True

    # Check if user already has saved resources in cache
    cached_resources: dict = obj_store.user_resources(username)

    # e.g. {"/jlewitt1/bert-preproc": "read"}
    cluster_access_level = cached_resources.get(cluster_uri)

    if cluster_access_level is None:
        # Reload from cache and check again
        obj_store.add_user_to_auth_cache(username, token)

        cached_resources: dict = obj_store.user_resources(username)
        cluster_access_level = cached_resources.get(cluster_uri)

    return cluster_access_level in [ResourceAccess.WRITE, ResourceAccess.READ]
