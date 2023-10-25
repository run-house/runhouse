import hashlib
import json
import logging
from typing import Union

import requests

from runhouse.globals import rns_client
from runhouse.rns.utils.api import load_resp_content

logger = logging.getLogger(__name__)


class AuthCache:
    AUTH_CACHE = {}

    @classmethod
    def get_user_resources(cls, token: str) -> dict:
        """Get resources associated with a particular user's token"""
        return cls.AUTH_CACHE.get(cls.hash_token(token), {})

    @classmethod
    def update_user_resources(cls, token: str, resources: dict):
        """Update server cache with a user's resources and access type"""
        cls.AUTH_CACHE[cls.hash_token(token)] = resources

    @classmethod
    def lookup_access_level(
        cls, token: str, resource_uri: str, retry=True
    ) -> Union[str, None]:
        resources: dict = cls.get_user_resources(cls.hash_token(token))
        if not resources and retry:
            cls.refresh_cache(token)

            # Try again after refreshing the cache
            return cls.lookup_access_level(token, resource_uri, retry=False)

        return resources.get(resource_uri)

    @classmethod
    def refresh_cache(cls, token):
        """Refresh the server cache with the latest resources and access levels"""
        resp = requests.get(
            f"{rns_client.api_server_url}/resource",
            headers={"Authorization": f"Bearer {token}"},
        )
        if resp.status_code != 200:
            raise Exception(f"Failed to refresh cache: {load_resp_content(resp)}")

        resp_data = json.loads(resp.content)
        all_resources: dict = {
            resource["name"]: resource["access_type"] for resource in resp_data["data"]
        }

        cls.update_user_resources(cls.hash_token(token), all_resources)
        logger.info(f"Updated cache for user with {len(all_resources)} resources")

    @staticmethod
    def hash_token(token: str) -> str:
        """Hash the user's token to avoid storing them in plain text on the cluster."""
        return hashlib.sha256(token.encode()).hexdigest()
