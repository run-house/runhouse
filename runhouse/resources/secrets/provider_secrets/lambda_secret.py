import copy
import os
from pathlib import Path

from typing import Optional

from runhouse.resources.secrets.provider_secrets.provider_secret import ProviderSecret


class LambdaSecret(ProviderSecret):
    _DEFAULT_CREDENTIALS_PATH = "~/.lambda_cloud/lambda_keys"
    _PROVIDER = "lambda"
    # _ENV_VARS = {"api_key": "LAMBDA_API_KEY"}

    @staticmethod
    def from_config(config: dict, dryrun: bool = False):
        return LambdaSecret(**config, dryrun=dryrun)

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

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w+") as f:
            f.write(f'api_key = {secrets["api_key"]}\n')

        return new_secret

    def _from_path(self, path: Optional[str] = None):
        path = path or self.path
        if path and os.path.exists(os.path.expanduser(path)):
            with open(os.path.expanduser(path), "r") as f:
                lines = f.readlines()
            for line in lines:
                split = line.split()
                if split[0] == "api_key":
                    api_key = split[-1]
                    break
            return {"api_key": api_key}
        return {}
