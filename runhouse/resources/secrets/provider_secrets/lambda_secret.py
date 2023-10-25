import copy
import os
from pathlib import Path

from typing import Optional

from runhouse.resources.blobs.file import File

from runhouse.resources.secrets.provider_secrets.provider_secret import ProviderSecret
from runhouse.resources.secrets.utils import _check_file_for_mismatches


class LambdaSecret(ProviderSecret):
    # values format: {"api_key": api_key}
    _DEFAULT_CREDENTIALS_PATH = "~/.lambda_cloud/lambda_keys"
    _PROVIDER = "lambda"

    @staticmethod
    def from_config(config: dict, dryrun: bool = False):
        return LambdaSecret(**config, dryrun=dryrun)

    def write(
        self,
        path: str = None,
        overwrite: bool = False,
    ):
        new_secret = copy.deepcopy(self)
        path = path or self.path
        if path:
            new_secret.path = path

        path = os.path.expanduser(path) if not isinstance(path, File) else path
        if _check_file_for_mismatches(
            path, self._from_path(path), self.values, overwrite
        ):
            return self

        values = self.values
        data = f'api_key = {values["api_key"]}\n'
        if isinstance(path, File):
            path.write(data, serialize=False, mode="w")
        else:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w+") as f:
                f.write(data)

        return new_secret

    def _from_path(self, path: Optional[str] = None):
        path = path or self.path
        lines = None
        if isinstance(path, File):
            if path.exists_in_system():
                lines = path.fetch(mode="r").split("\n")
        elif path and os.path.exists(os.path.expanduser(path)):
            with open(os.path.expanduser(path), "r") as f:
                lines = f.readlines()
        if lines:
            for line in lines:
                split = line.split()
                if split[0] == "api_key":
                    api_key = split[-1]
                    return {"api_key": api_key}
        return {}
