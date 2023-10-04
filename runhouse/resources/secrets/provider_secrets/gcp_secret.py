import copy
import json
import os
from pathlib import Path

from typing import Optional

from runhouse.resources.secrets.provider_secrets.provider_secret import ProviderSecret


class GCPSecret(ProviderSecret):
    _DEFAULT_CREDENTIALS_PATH = "~/.config/gcloud/application_default_credentials.json"
    _PROVIDER = "gcp"
    _ENV_VARS = {
        "client_id": "CLIENT_ID",
        "client_secret": "CLIENT_SECRET",
    }

    @staticmethod
    def from_config(config: dict, dryrun: bool = False):
        return GCPSecret(**config, dryrun=dryrun)

    def write(
        self,
        path: str = None,
    ):
        new_secret = copy.deepcopy(self)
        if path:
            new_secret.path = path

        path = path or self.path
        path = os.path.expanduser(path)
        secrets = self.secrets

        config = {}
        if Path(path).exists():
            with open(path, "r") as config_file:
                config = json.load(config_file)
        for key in secrets.keys():
            config[key] = secrets[key]

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w+") as f:
            json.dump(config, f, indent=4)

        return new_secret

    def _from_path(self, path: Optional[str] = None):
        path = path or self.path
        if path and os.path.exists(os.path.expanduser(path)):
            with open(os.path.expanduser(path), "r") as config_file:
                config = json.load(config_file)
            client_id = config["client_id"]
            client_secret = config["client_secret"]

            return {
                "client_id": client_id,
                "client_secret": client_secret,
            }
        return {}
