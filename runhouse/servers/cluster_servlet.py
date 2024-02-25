import logging
from typing import Any, Dict, List, Optional, Set, Union

from runhouse.resources.hardware import load_cluster_config_from_file
from runhouse.servers.http.auth import AuthCache

logger = logging.getLogger(__name__)


class ClusterServlet:
    def __init__(
        self, cluster_config: Optional[Dict[str, Any]] = None, *args, **kwargs
    ):

        # We do this here instead of at the start of the HTTP Server startup
        # because someone can be running `HTTPServer()` standalone in a test
        # and still want an initialized cluster config in the servlet.
        if not cluster_config:
            cluster_config = load_cluster_config_from_file()

        self.cluster_config: Optional[Dict[str, Any]] = (
            cluster_config if cluster_config else {}
        )
        self._initialized_env_servlet_names: Set[str] = set()
        self._key_to_env_servlet_name: Dict[Any, str] = {}
        self._auth_cache: AuthCache = AuthCache()

    ##############################################
    # Cluster config state storage methods
    ##############################################
    def get_cluster_config(self) -> Dict[str, Any]:
        return self.cluster_config

    def set_cluster_config(self, cluster_config: Dict[str, Any]):
        self.cluster_config = cluster_config

    def set_cluster_config_value(self, key: str, value: Any):
        self.cluster_config[key] = value

    ##############################################
    # Auth cache internal functions
    ##############################################
    def add_user_to_auth_cache(self, username, token, refresh_cache=True):
        self._auth_cache.add_user(username, token, refresh_cache)

    def resource_access_level(
        self, username: str, resource_uri: str
    ) -> Union[str, None]:
        return self._auth_cache.lookup_access_level(username, resource_uri)

    def user_resources(self, username: str) -> dict:
        return self._auth_cache.get_user_resources(username)

    def has_resource_access(self, username: str, resource_uri=None) -> bool:
        """Checks whether user has read or write access to a given module saved on the cluster."""
        from runhouse.rns.utils.api import ResourceAccess

        cluster_uri = self.cluster_config["name"]
        cluster_access = self.resource_access_level(username, cluster_uri)
        if cluster_access == ResourceAccess.WRITE:
            # if user has write access to cluster will have access to all resources
            return True

        if resource_uri is None and cluster_access not in [
            ResourceAccess.WRITE,
            ResourceAccess.READ,
        ]:
            # If module does not have a name, must have access to the cluster
            return False

        resource_access_level = self.resource_access_level(username, resource_uri)
        if resource_access_level not in [ResourceAccess.WRITE, ResourceAccess.READ]:
            return False

        return True

    def clear_auth_cache(self, username: str = None):
        self._auth_cache.clear_cache(username)

    ##############################################
    # Key to servlet where it is stored mapping
    ##############################################
    def mark_env_servlet_name_as_initialized(self, env_servlet_name: str):
        self._initialized_env_servlet_names.add(env_servlet_name)

    def is_env_servlet_name_initialized(self, env_servlet_name: str) -> bool:
        return env_servlet_name in self._initialized_env_servlet_names

    def get_all_initialized_env_servlet_names(self) -> Set[str]:
        return self._initialized_env_servlet_names

    def get_key_to_env_servlet_name_dict_keys(self) -> List[Any]:
        return list(self._key_to_env_servlet_name.keys())

    def get_key_to_env_servlet_name_dict(self) -> Dict[Any, str]:
        return self._key_to_env_servlet_name

    def get_env_servlet_name_for_key(self, key: Any) -> str:
        return self._key_to_env_servlet_name.get(key, None)

    def put_env_servlet_name_for_key(self, key: Any, env_servlet_name: str):
        if not self.is_env_servlet_name_initialized(env_servlet_name):
            raise ValueError(
                f"Env servlet name {env_servlet_name} not initialized, and you tried to mark a resource as in it."
            )
        self._key_to_env_servlet_name[key] = env_servlet_name

    def pop_env_servlet_name_for_key(self, key: Any, *args) -> str:
        # *args allows us to pass default or not
        return self._key_to_env_servlet_name.pop(key, *args)

    def clear_key_to_env_servlet_name_dict(self):
        self._key_to_env_servlet_name = {}

    ##############################################
    # Remove Env Servlet
    ##############################################
    def remove_env_servlet_name(self, env_servlet_name: str):
        self._initialized_env_servlet_names.remove(env_servlet_name)
