import copy
import os
from pathlib import Path

from typing import Dict, Union

from runhouse.resources.blobs.file import File

from runhouse.resources.secrets.provider_secrets.provider_secret import ProviderSecret
from runhouse.resources.secrets.utils import _check_file_for_mismatches


class LambdaSecret(ProviderSecret):
    """
    .. note::
            To create a LambdaSecret, please use the factory method :func:`provider_secret` with ``provider="lambda"``.
    """

    # values format: {"api_key": api_key}
    _DEFAULT_CREDENTIALS_PATH = "~/.lambda_cloud/lambda_keys"
    _PROVIDER = "lambda"

    @staticmethod
    def from_config(config: dict, dryrun: bool = False, _resolve_children: bool = True):
        return LambdaSecret(**config, dryrun=dryrun)

    def _write_to_file(
        self, path: Union[str, File], values: Dict = None, overwrite: bool = False
    ):
        new_secret = copy.deepcopy(self)
        if not _check_file_for_mismatches(
            path, self._from_path(path), values, overwrite
        ):
            data = f'api_key = {values["api_key"]}\n'
            if isinstance(path, File):
                path.write(data, serialize=False, mode="w")
            else:
                full_path = os.path.expanduser(path)
                Path(full_path).parent.mkdir(parents=True, exist_ok=True)
                with open(full_path, "w+") as f:
                    f.write(data)
                new_secret._add_to_rh_config(path)

        new_secret._values = None
        new_secret.path = path
        return new_secret

    def _from_path(self, path: Union[str, File]):
        lines = None
        if isinstance(path, File):
            if path.exists_in_system():
                lines = path.fetch(mode="r", deserialize=False).split("\n")
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
