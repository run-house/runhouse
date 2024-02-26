import json
import logging
from typing import Optional, Union

from runhouse.globals import rns_client
from runhouse.rns.utils.api import load_resp_content, ResourceAccess
from runhouse.servers.http.http_utils import username_from_token

logger = logging.getLogger(__name__)


class AuthCache:
    # Maps a user's token to all the resources they have access to
    CACHE = {}
    USERNAMES = {}

    @classmethod
    def get_user_resources(cls, token: str) -> dict:
        """Get resources associated with a particular token"""
        return cls.CACHE.get(token, {})

    @classmethod
    def get_username(cls, token: str) -> Optional[str]:
        """Get username associated with a particular token"""
        return cls.USERNAMES.get(token)

    @classmethod
    def lookup_access_level(cls, token: str, resource_uri: str) -> Union[str, None]:
        resources: dict = cls.get_user_resources(token)
        return resources.get(resource_uri)

    @classmethod
    def add_user(cls, token, refresh_cache=True):
        """Refresh the server cache with the latest resources and access levels for a particular token"""
        if token is None:
            return

        if not refresh_cache and token in cls.CACHE:
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

        username = username_from_token(token)
        if username is None:
            raise ValueError("Failed to find Runhouse user from provided token.")
        cls.USERNAMES[token] = username

        resp_data = json.loads(resp.content)
        # Support access_level and access_type for BC
        all_resources: dict = {
            resource["name"]: resource.get("access_level")
            or resource.get("access_type")
            for resource in resp_data["data"]
        }
        # Update server cache with a user's resources and access type
        cls.CACHE[token] = all_resources

    def clear_cache(self, token: str = None):
        """Clear the server cache, If a token is specified, clear the cache for that particular user only"""
        if token is None:
            self.CACHE = {}
            self.USERNAMES = {}
        else:
            self.CACHE.pop(token, None)
            self.USERNAMES.pop(token, None)


def verify_cluster_access(
    cluster_uri: str,
    token: str,
) -> bool:
    """Checks whether the user has access to the cluster.
    Note: A user with write access to the cluster or a cluster owner will have access to all other resources on
    the cluster by default."""
    from runhouse.globals import configs, obj_store

    # The logged-in user always has full access to the cluster. This is especially important if they flip on
    # Den Auth without saving the cluster. We may need to generate a subtoken here to check.
    if configs.token:
        if configs.token == token:
            return True
        if (
            cluster_uri
            and rns_client.cluster_token(configs.token, cluster_uri) == token
        ):
            return True

    # Check if user already has saved resources in cache
    cached_resources: dict = obj_store.user_resources(token)

    # e.g. {"/jlewitt1/bert-preproc": "read"}
    cluster_access_level = cached_resources.get(cluster_uri)

    if cluster_access_level is None:
        # Reload from cache and check again
        obj_store.add_user_to_auth_cache(token)

        cached_resources: dict = obj_store.user_resources(token)
        cluster_access_level = cached_resources.get(cluster_uri)

    return cluster_access_level in [ResourceAccess.WRITE, ResourceAccess.READ]
