import copy
import json
import os
from pathlib import Path

from typing import Optional

from runhouse.resources.blobs.file import File

from runhouse.resources.secrets.provider_secrets.provider_secret import ProviderSecret
from runhouse.resources.secrets.utils import _check_file_for_mismatches


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
        overwrite: bool = False,
    ):
        new_secret = copy.deepcopy(self)
        path = path or self.path
        if path:
            new_secret.path = path

        if not isinstance(path, File):
            path = os.path.expanduser(path)

        values = self.values
        if _check_file_for_mismatches(path, self._from_path(path), values, overwrite):
            return self

        if isinstance(path, File):
            data = json.dumps(values, indent=4)
            path.write(data, serialize=False, mode="w")
        else:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w+") as f:
                json.dump(values, f, indent=4)

        return new_secret

    def _from_path(self, path: Optional[str] = None):
        path = path or self.path
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
