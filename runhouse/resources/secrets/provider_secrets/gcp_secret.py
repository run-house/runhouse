import copy
import json
import os
from pathlib import Path

from typing import Dict, Union

from runhouse.resources.blobs.file import File

from runhouse.resources.secrets.provider_secrets.provider_secret import ProviderSecret
from runhouse.resources.secrets.utils import _check_file_for_mismatches


class GCPSecret(ProviderSecret):
    _PROVIDER = "gcp"
    _DEFAULT_CREDENTIALS_PATH = "~/.config/gcloud/application_default_credentials.json"
    _DEFAULT_ENV_VARS = {
        "client_id": "CLIENT_ID",
        "client_secret": "CLIENT_SECRET",
    }

    @staticmethod
    def from_config(config: dict, dryrun: bool = False):
        return GCPSecret(**config, dryrun=dryrun)

    def _write_to_file(
        self, path: Union[str, File], values: Dict = None, overwrite: bool = False
    ):
        path = os.path.expanduser(path) if not isinstance(path, File) else path
        if _check_file_for_mismatches(path, self._from_path(path), values, overwrite):
            return self

        new_secret = copy.deepcopy(self)
        new_secret._values = None
        new_secret.path = path

        if isinstance(path, File):
            data = json.dumps(values, indent=4)
            path.write(data, serialize=False, mode="w")
        else:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w+") as f:
                json.dump(values, f, indent=4)
            new_secret._add_to_rh_config(path)

        return new_secret

    def _from_path(self, path: Union[str, File]):
        config = {}
        if isinstance(path, File):
            if not path.exists_in_system():
                return {}
            contents = path.fetch(mode="r")
            config = json.laods(contents)
        elif path and os.path.exists(os.path.expanduser(path)):
            with open(os.path.expanduser(path), "r") as config_file:
                config = json.load(config_file)
        return config
