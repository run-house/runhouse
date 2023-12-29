import logging
from pathlib import Path

from runhouse.servers.http.auth import AuthCache

logger = logging.getLogger(__name__)


class ClusterServlet:
    LOGS_DIR = ".rh/logs"
    RH_LOGFILE_PATH = Path.home() / LOGS_DIR

    def __init__(self, cluster_config=None, *args, **kwargs):
        self.cluster_config = cluster_config

        self._env_for_key = {}
        self._auth_cache = AuthCache()
        self._cluster_connections = {}

    # Not properties because ray doesn't handle properties so elegantly
    def cluster_config(self):
        return self.cluster_config

    # TODO broadcast updates to all obj stores via servlets
    def set_cluster_config(self, cluster_config):
        self.cluster_config = cluster_config

    #### Auth Methods ####

    def add_user(self, token, refresh_cache=True):
        self._auth_cache.add_user(token, refresh_cache)

    def resource_access_level(self, token_hash: str, resource_uri: str):
        return self._auth_cache.lookup_access_level(token_hash, resource_uri)

    def user_resources(self, token_hash: str):
        return self._auth_cache.get_user_resources(token_hash)

    def has_resource_access(self, token_hash: str, resource_uri=None) -> bool:
        """Checks whether user has read or write access to a given module saved on the cluster."""
        from runhouse.rns.utils.api import ResourceAccess

        if token_hash is None:
            # If no token is provided assume no access
            return False

        cluster_uri = self.cluster_config["name"]
        cluster_access = self.resource_access_level(token_hash, cluster_uri)
        if cluster_access == ResourceAccess.WRITE:
            # if user has write access to cluster will have access to all resources
            return True

        if resource_uri is None and cluster_access not in [
            ResourceAccess.WRITE,
            ResourceAccess.READ,
        ]:
            # If module does not have a name, must have access to the cluster
            return False

        resource_access_level = self.resource_access_level(token_hash, resource_uri)
        if resource_access_level not in [ResourceAccess.WRITE, ResourceAccess.READ]:
            return False

        return True

    def clear_auth_cache(self, token_hash: str = None):
        self._auth_cache.clear_cache(token_hash)

    #### Env-Key Lookup Methods ####

    def keys(self, return_envs=False):
        # Return keys across the cluster, not only in this process
        return self._env_for_key if return_envs else list(self._env_for_key.keys())

    def get_env(self, key):
        return self._env_for_key.get(key, None)

    def put_env(self, key, value):
        self._env_for_key[key] = value

    def rename_key(self, old_key, new_key, *args):
        # *args allows us to pass default or not
        self._env_for_key[new_key] = self._env_for_key.pop(old_key, *args)

    def pop_env(self, key: str, *args):
        # *args allows us to pass default or not
        self._env_for_key.pop(key, *args)

    def clear_env(self):
        self._env_for_key = {}

    def contains(self, key: str):
        return key in self._env_for_key

    def get_logfiles(self, key: str, log_type=None):
        # TODO remove
        # Info on ray logfiles: https://docs.ray.io/en/releases-2.2.0/ray-observability/ray-logging.html#id1
        if self.contains(key):
            # Logs are like: `.rh/logs/key.[out|err]`
            key_logs_path = Path(self.RH_LOGFILE_PATH) / key
            glob_pattern = (
                "*.out"
                if log_type == "stdout"
                else "*.err"
                if log_type == "stderr"
                else "*.[oe][ur][tr]"
            )
            return [str(f.absolute()) for f in key_logs_path.glob(glob_pattern)]
        else:
            return None

