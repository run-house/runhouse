import copy
import json
import os
from pathlib import Path

from typing import Dict

from runhouse.resources.secrets.provider_secrets.provider_secret import ProviderSecret
from runhouse.resources.secrets.utils import _check_file_for_mismatches


class GCPSecret(ProviderSecret):
    """
    .. note::
            To create a GCPSecret, please use the factory method :func:`provider_secret` with ``provider="gcp"``.
    """

    _PROVIDER = "gcp"
    _DEFAULT_CREDENTIALS_PATH = "~/.config/gcloud/application_default_credentials.json"
    _DEFAULT_ENV_VARS = {
        "client_id": "CLIENT_ID",
        "client_secret": "CLIENT_SECRET",
    }

    @staticmethod
    def from_config(config: dict, dryrun: bool = False, _resolve_children: bool = True):
        return GCPSecret(**config, dryrun=dryrun)

    def _write_to_file(self, path: str, values: Dict = None, overwrite: bool = False):
        new_secret = copy.deepcopy(self)
        if not _check_file_for_mismatches(
            path, self._from_path(path), values, overwrite
        ):
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w+") as f:
                json.dump(values, f, indent=4)
            new_secret._add_to_rh_config(path)

        new_secret._values = None
        new_secret.path = path
        return new_secret

    def _from_path(self, path: str = None):
        config = {}
        if path and os.path.exists(os.path.expanduser(path)):
            with open(os.path.expanduser(path), "r") as config_file:
                config = json.load(config_file)
        return config
