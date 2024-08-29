import contextvars
import copy
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Optional

import requests
import yaml

from runhouse.logger import get_logger

from runhouse.rns.utils.api import read_resp_data, to_bool


req_ctx = contextvars.ContextVar("rh_ctx", default={})

logger = get_logger(__name__)


class Defaults:
    """Class to handle defaults for Runhouse. Defaults are stored in a json file in the user's home directory."""

    USER_ENDPOINT = "user"
    GROUP_ENDPOINT = "group"
    CONFIG_PATH = Path("~/.rh/config.yaml").expanduser()
    # TODO [DG] default sub-dicts for various resources (e.g. defaults.get('cluster').get('resource_type'))
    BASE_DEFAULTS = {
        "default_folder": "~",
        "default_provider": "cheapest",
        "default_autostop": -1,
        "use_spot": False,
        "use_local_configs": True,
        "disable_data_collection": False,
        "use_rns": False,
        "api_server_url": "https://api.run.house",
        "dashboard_url": "https://run.house",
        "autosave": True,
    }

    def __init__(self):
        self._token = None
        self._username = None
        self._default_folder = None
        self._defaults_cache = defaultdict(dict)
        self._simulate_logged_out = False
        self._use_caller_token = False

    @property
    def token(self):
        if self._simulate_logged_out:
            return None

        # The token can be manually overriden in Python when we are logging in a test user and such
        if self._token:
            return self._token

        if self._use_caller_token:
            req_ctx_obj = req_ctx.get()
            if req_ctx_obj:
                return req_ctx_obj.token

        # If nothing was set in req_ctx, we default to the environment variable
        if os.environ.get("RH_TOKEN"):
            return os.environ.get("RH_TOKEN")

        # Finally, we check the defaults cache, which is loaded from the config.yaml file
        if "token" in self.defaults_cache:
            return self.defaults_cache["token"]

        return None

    @token.setter
    def token(self, value):
        self._token = value

    @property
    def username(self):
        if self._simulate_logged_out:
            return None

        # This is not to "cache" the username, but rather to allow us to manually override it in python
        if self._username:
            return self._username
        if os.environ.get("RH_USERNAME"):
            self._username = os.environ.get("RH_USERNAME")
            return self._username
        if "username" in self.defaults_cache:
            self._username = self.defaults_cache["username"]
            return self._username
        return None

    @username.setter
    def username(self, value):
        self._username = value

    @property
    def default_folder(self):
        if self._simulate_logged_out:
            return self.BASE_DEFAULTS["default_folder"]

        # This is not to "cache" the default_folder, but rather to allow us to manually override it in python
        if self._default_folder:
            return self._default_folder
        if "default_folder" in self.defaults_cache:
            self._default_folder = self.defaults_cache["default_folder"]
            return self._default_folder
        if self.username:
            self._default_folder = "/" + self.username
            return self._default_folder
        return self.BASE_DEFAULTS["default_folder"]

    @default_folder.setter
    def default_folder(self, value):
        self._default_folder = value

    @property
    def defaults_cache(self):
        if self._simulate_logged_out:
            return {}

        if not self._defaults_cache:
            self._defaults_cache = self.load_defaults_from_file()
        return self._defaults_cache

    @defaults_cache.setter
    def defaults_cache(self, value: Dict):
        self._defaults_cache = value

    def load_defaults_from_file(self, config_path: Optional[str] = None) -> Dict:
        config_path = config_path or self.CONFIG_PATH
        config = {}
        if Path(config_path).exists():
            with open(config_path, "r") as stream:
                config = yaml.safe_load(stream)
            logger.info(f"Loaded Runhouse config from {config_path}")

        return config or {}

    @property
    def request_headers(self) -> dict:
        """Base request headers used to make requests to Runhouse Den."""
        return {"Authorization": f"Bearer {self.token}"} if self.token else {}

    def upload_defaults(
        self,
        defaults: Optional[Dict] = None,
        headers: Optional[Dict] = None,
        entity: Optional[str] = "user",
    ):
        """Upload defaults into rns. If defaults is None, upload the defaults from the local config file,
        `~/.rh/config.yaml."""
        if not defaults:
            defaults = self.defaults_cache
            self.save_defaults()
        to_upload = copy.deepcopy(defaults)
        # We don't need to save these
        to_upload.pop("token", None)
        to_upload.pop("username", None)
        to_upload.pop("secrets", None)

        endpoint = (
            self.USER_ENDPOINT
            if entity == "user"
            else f"{self.GROUP_ENDPOINT}/{entity}"
        )
        uri = f'{self.get("api_server_url")}/{endpoint}/config'
        resp = requests.put(
            uri,
            data=json.dumps(to_upload),
            headers=headers or self.request_headers,
        )
        if resp.status_code != 200:
            raise Exception(
                f"Received [{resp.status_code}] from Den PUT '{uri}': Failed to update defaults for {entity}."
            )
        logger.info(f"Uploaded defaults for {entity} to rns.")

    def download_defaults(
        self, headers: Optional[Dict] = None, entity: Optional[str] = "user"
    ) -> Dict:
        """Get defaults for user or group."""
        endpoint = (
            self.USER_ENDPOINT
            if entity == "user"
            else f"{self.GROUP_ENDPOINT}/{entity}"
        )
        headers = headers or self.request_headers
        uri = f'{self.get("api_server_url")}/{endpoint}'
        resp = requests.get(uri, headers=headers)
        if resp.status_code != 200:
            raise Exception(
                f"Received [{resp.status_code}] from Den GET '{uri}': Failed to download defaults for {entity}."
            )
        resp_data: dict = read_resp_data(resp)
        raw_defaults = resp_data.get("config", {})
        raw_defaults["username"] = resp_data.get("username")
        raw_defaults["token"] = headers.get("Authorization")[7:]
        raw_defaults["default_folder"] = (
            "/" + raw_defaults["username"]
            if raw_defaults.get("default_folder") in ["~", None]
            else raw_defaults.get("default_folder")
        )
        formatted = {k: to_bool(v) for k, v in raw_defaults.items()}

        return formatted

    def save_defaults(
        self, defaults: Optional[Dict] = None, config_path: Optional[str] = None
    ):
        config_path = Path(config_path or self.CONFIG_PATH)
        defaults = defaults or self.defaults_cache
        if not defaults:
            return

        if not config_path.exists():
            config_path.parent.mkdir(parents=True, exist_ok=True)

        with config_path.open("w") as stream:
            yaml.safe_dump(defaults, stream)

    def download_and_save_defaults(
        self,
        headers: Optional[Dict] = None,
        entity: Optional[str] = "user",
        config_path: Optional[str] = None,
    ):
        """Download defaults from rns and save them to the local config file."""
        defaults = self.download_defaults(headers=headers, entity=entity)
        # Note: downloaded defaults take priority over local defaults
        self.set_many(defaults, config_path=config_path)

    def set(self, key: str, value: Any, config_path: Optional[str] = None):
        self.defaults_cache[key] = value
        self.save_defaults(config_path=config_path)

    def set_nested(self, key: str, value: Any, config_path: Optional[str] = None):
        """Set a config key that has multiple key/value pairs"""
        if not self.defaults_cache.get(key):
            self.defaults_cache.setdefault(key, {})
        self.defaults_cache[key].update(value)
        self.save_defaults(config_path=config_path)

    def set_many(self, key_value_pairs: Dict, config_path: Optional[str] = None):
        self.defaults_cache.update(key_value_pairs)
        self.save_defaults(config_path=config_path)

    # TODO [DG] allow hierarchical defaults from folders and groups
    def get(self, key: str, alt: Any = None) -> Any:
        # Prioritize env vars
        env_var = os.getenv(key.upper())
        if env_var is not None:
            return env_var

        res = self.defaults_cache.get(key, alt)
        if res is None and key in self.BASE_DEFAULTS:
            res = self.BASE_DEFAULTS[key]
        return res

    def delete(self, key: str):
        """Remove a specific key from the config"""
        self.defaults_cache.pop(key, None)
        self.save_defaults()

    def delete_provider(self, provider: str):
        """Remove a specific provider from the config secrets."""
        if self.defaults_cache.get("secrets"):
            self.defaults_cache.get("secrets").pop(provider, None)
        self.save_defaults()

    def delete_defaults(self, config_path: Optional[str] = None):
        """Delete the defaults file entirely"""
        config_path = Path(config_path or self.CONFIG_PATH)
        try:
            Path(config_path).unlink(missing_ok=True)
            logger.info(f"Deleted config file from path: {config_path}")
        except OSError:
            raise Exception(f"Failed to delete config file from path {config_path}")

    def disable_data_collection(self):
        self.set("disable_data_collection", True)
        os.environ["DISABLE_DATA_COLLECTION"] = "True"

    def data_collection_enabled(self) -> bool:
        """Checks whether to enable data collection, based on values set in the local ~/.rh config or as an env var."""
        if self.get("disable_data_collection") is True:
            return False
        if os.getenv("DISABLE_DATA_COLLECTION", "False").lower() in ("true", "1"):
            return False

        return True
