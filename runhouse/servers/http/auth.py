import logging
from typing import Optional, Union

from runhouse.globals import rns_client
from runhouse.rns.utils.api import load_resp_content, ResourceAccess
from runhouse.servers.http.http_utils import (
    get_username_from_cluster_token,
    username_from_token,
)

logger = logging.getLogger(__name__)


class AuthCache:
    # Maps a user's token to all the resources they have access to
    def __init__(self, cluster_config: dict):
        self.CACHE = {}
        self.USERNAMES = {}
        self.cluster_config = cluster_config

    def get_username(self, token: str) -> Optional[str]:
        """Get username associated with a particular token"""
        if token not in self.USERNAMES:
            username = username_from_token(token)
            if username is not None:
                self.USERNAMES[token] = username

        return self.USERNAMES.get(token)

    def lookup_access_level(
        self, token: str, resource_uri: str, refresh_cache=True
    ) -> Union[str, None]:
        """Get the access level of a particular resource for a user"""
        if token is None or resource_uri is None:
            return

        # Also add this user to the username cache
        self.get_username(token)

        if (token, resource_uri) in self.CACHE and not refresh_cache:
            return self.CACHE[(token, resource_uri)]

        if resource_uri.startswith("/"):
            resource_uri_to_send = resource_uri[1:].replace("/", ":")
        else:
            resource_uri_to_send = resource_uri.replace("/", ":")

        api_server_url = (
            self.cluster_config.get("api_server_url") or rns_client.api_server_url
        )
        uri = f"{api_server_url}/resource/{resource_uri_to_send}"
        resp = rns_client.session.get(
            uri,
            headers={"Authorization": f"Bearer {token}"},
        )

        if resp.status_code == 404:
            logger.error(
                f"Received [{resp.status_code}] from Den GET '{uri}': Resource not found: {load_resp_content(resp)}"
            )
            return

        if resp.status_code != 200:
            logger.error(
                f"Received [{resp.status_code}] from Den GET '{uri}': Failed to load access level for resource: {load_resp_content(resp)}"
            )
            return

        self.CACHE[(token, resource_uri)] = resp.json()["data"]["access_level"]

        return self.CACHE[(token, resource_uri)]

    def clear_cache(self, token: str = None):
        """Clear the server cache, If a token is specified, clear the cache for that particular user only"""
        if token is None:
            self.CACHE = {}
            self.USERNAMES = {}
        else:
            self.CACHE.pop(token, None)
            self.USERNAMES.pop(token, None)


async def averify_cluster_access(
    cluster_uri: str,
    token: str,
) -> bool:
    """Checks whether the user has access to the cluster.
    Note: A user with write access to the cluster or a cluster owner will have access to all other resources on
    the cluster by default."""
    from runhouse.globals import configs, obj_store

    # The logged-in user always has full access to the cluster. This is especially important if they flip on
    # Den Auth without saving the cluster.
    username = get_username_from_cluster_token(token)
    cluster_token = configs.load_cluster_token_from_file(username=username)
    if cluster_token:
        # If cluster token is saved down in file: "cluster_owners.yaml"
        return True

    cluster_access_level = await obj_store.aresource_access_level(token, cluster_uri)

    return cluster_access_level in [ResourceAccess.WRITE, ResourceAccess.READ]
