import os
from enum import Enum


class EnvOptions(Enum):
    IS_DEVELOPER = "RUNHOUSE_DEV"
    DISABLE_USAGE_COLLECTION = "RAY_USAGE_STATS_ENABLED"

    def disable_usage_collection(self):
        """Checks whether to turn off usage collection. Returns `True` if usage collection should be disabled."""
        # https://docs.ray.io/en/latest/cluster/usage-stats.html
        if os.getenv(self.value) == "0":
            return True

        try:
            from runhouse import Secrets

            ray_config: dict = Secrets.read_json_file("~/.ray/config.json")
            if ray_config.get("usage_stats") is False:
                # Turn off usage collection if explicitly set to False in the config
                return True
        except FileNotFoundError:
            pass

        return False

    def developer_mode(self):
        """Checks whether environment is dev or prod."""
        from runhouse import configs

        env_val = os.getenv(self.value, "False").lower() in ("true", "1")
        api_server_url = configs.get("api_server_url")
        if env_val or (
            api_server_url is not None
            and api_server_url != configs.BASE_DEFAULTS["api_server_url"]
        ):
            return True
        return False
