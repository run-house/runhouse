import json
import logging
import requests
import yaml
from typing import Dict, Optional, Any
from pathlib import Path
import copy

from runhouse.rns.api_utils.utils import read_response_data, to_bool

from runhouse.logger import LOGGING_CONFIG

# Configure the logger once
logging.config.dictConfig(LOGGING_CONFIG)

logger = logging.getLogger(__name__)


class Defaults:
    """Class to handle defaults for Runhouse. Defaults are stored in a json file in the user's home directory."""

    USER_ENDPOINT = 'user/'
    GROUP_ENDPOINT = 'group/'
    CONFIG_PATH = Path('~/.rh/config.yaml').expanduser()
    # TODO [DG] default sub-dicts for various resources (e.g. defaults.get('cluster').get('type'))
    BASE_DEFAULTS = {'default_folder': '/default',
                     'default_provider': 'cheapest',
                     'default_autostop': -1,
                     'use_spot': False,
                     'use_local_configs': True,
                     'use_rns': False,
                     'api_server_url': "https://api.run.house"
                     }

    def __init__(self):
        self.defaults_cache = self.load_defaults_from_file(add_base_defaults=True)

    def load_defaults_from_file(self,
                                config_path: Optional[str] = None,
                                add_base_defaults: bool = True) -> Dict:
        config_path = config_path or self.CONFIG_PATH
        config = {}
        if Path(config_path).exists():
            with open(config_path, 'r') as stream:
                config = yaml.safe_load(stream)
        if add_base_defaults:
            config = self.merge_with_base_defaults(config)

        self.defaults_cache = copy.deepcopy(config)

        logging.info(f'Loaded Runhouse config from {config_path}')

        return config or {}

    # TODO turn this into a general purpose "reconcile n configs with differing priorities" function
    @classmethod
    def merge_with_base_defaults(cls, defaults: Dict) -> Dict:
        # We need to update defaults with config so existing config overwrites the defaults
        merged = copy.deepcopy(cls.BASE_DEFAULTS)
        merged.update(defaults)
        return merged

    @property
    def request_headers(self):
        return {"Authorization": f"Bearer {self.get('token')}"}

    def upload_defaults(self,
                        defaults: Optional[Dict] = None,
                        headers: Optional[Dict] = None,
                        entity: Optional[str] = 'user'):
        """Upload defaults into rns. If defaults is None, upload the defaults from the local config file,
        `~/.rh/config.yaml."""
        if not defaults:
            defaults = self.load_defaults_from_file(add_base_defaults=False)
        # We don't need to save these
        del defaults['token']
        del defaults['username']

        endpoint = self.USER_ENDPOINT if entity == 'user' else self.GROUP_ENDPOINT + f'/{entity}'
        resp = requests.put(f'{self.get("api_server_url")}/{endpoint}/config',
                            data=json.dumps(defaults),
                            headers=headers or self.request_headers)
        if resp.status_code != 200:
            raise Exception(f'Failed to update defaults for {entity}, received status code {resp.status_code}')
        logger.info(f'Uploaded defaults for {entity} to rns.')

    def download_defaults(self,
                          headers: Optional[Dict] = None,
                          entity: Optional[str] = 'user') -> Dict:
        """ Get defaults for user or group."""
        endpoint = self.USER_ENDPOINT if entity == 'user' else self.GROUP_ENDPOINT + f'/{entity}'
        headers = headers or self.request_headers
        resp = requests.get(f'{self.get("api_server_url")}/{endpoint}',
                            headers=headers)
        if resp.status_code != 200:
            raise Exception(f'Failed to download defaults for {entity}, received status code {resp.status_code}')
        resp_data: dict = read_response_data(resp)
        raw_defaults = resp_data.get('config', {})
        raw_defaults["username"] = resp_data.get('username')
        raw_defaults["token"] = headers.get('Authorization')[7:]
        raw_defaults['default_folder'] = raw_defaults["username"] if \
            raw_defaults['default_folder'] == 'default/' else raw_defaults['default_folder']
        formatted = {k: to_bool(v) for k, v in raw_defaults.items()}

        self.defaults_cache = copy.deepcopy(formatted)

        return formatted

    @classmethod
    def save_defaults(cls, defaults: Dict, config_path: Optional[str] = None):
        config_path = Path(config_path or cls.CONFIG_PATH)

        if not config_path.exists():
            config_path.parent.mkdir(parents=True, exist_ok=True)

        with config_path.open('w') as stream:
            yaml.dump(defaults, stream)

    def download_and_save_defaults(self,
                                   headers: Optional[Dict] = None,
                                   entity: Optional[str] = 'user',
                                   merge_with_existing: bool = True,
                                   merge_with_base_defaults: bool = True,
                                   upload_merged: bool = True,
                                   config_path: Optional[str] = None):
        """Download defaults from rns and save them to the local config file."""
        defaults = self.download_defaults(headers=headers, entity=entity)
        if merge_with_existing:
            # Note: local defaults take precedence over downloaded defaults
            existing = self.load_defaults_from_file(config_path=config_path,
                                                    add_base_defaults=False)
            defaults.update(existing)
        if merge_with_base_defaults:
            defaults = self.merge_with_base_defaults(defaults)
        self.save_defaults(defaults, config_path=config_path)
        if (merge_with_existing or merge_with_base_defaults) and upload_merged:
            self.upload_defaults(defaults, headers=headers, entity=entity)

        self.defaults_cache = copy.deepcopy(defaults)

    def set(self, key: str, value: Any):
        self.defaults_cache[key] = value
        self.save_defaults(self.defaults_cache)

    # TODO [DG] allow hierarchical defaults from folders and groups
    def get(self, key: str, alt: Any = None) -> Any:
        return self.defaults_cache.get(key, alt)

