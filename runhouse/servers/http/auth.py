import hashlib
import logging

logger = logging.getLogger(__name__)


class ServerCache:
    # TODO: Turn this into a Ray actor to handle token and resource auth
    AUTH_CACHE = {}

    @classmethod
    def get_resources(cls, token: str) -> dict:
        """Get resources associated with a particular user's token"""
        return cls.AUTH_CACHE.get(cls.hash_token(token), {})

    @classmethod
    def put_resources(cls, token: str, resources: dict):
        """Update server cache with a user's resources and access type"""
        cls.AUTH_CACHE[cls.hash_token(token)] = resources

    @staticmethod
    def hash_token(token: str) -> str:
        """Hash the user's token to avoid storing them in plain text on the cluster."""
        return hashlib.sha256(token.encode()).hexdigest()
