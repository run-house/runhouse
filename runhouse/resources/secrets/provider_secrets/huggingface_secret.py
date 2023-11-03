import copy
import os
from pathlib import Path

from typing import Dict, Union

from runhouse.resources.blobs.file import File
from runhouse.resources.secrets.provider_secrets.provider_secret import ProviderSecret
from runhouse.resources.secrets.utils import _check_file_for_mismatches


class HuggingFaceSecret(ProviderSecret):
    # values format: {"token": hf_token}
    _DEFAULT_CREDENTIALS_PATH = "~/.cache/huggingface/token"
    _PROVIDER = "huggingface"

    @staticmethod
    def from_config(config: dict, dryrun: bool = False):
        return HuggingFaceSecret(**config, dryrun=dryrun)

    def _write_to_file(
        self, path: Union[str, File], values: Dict = None, overwrite: bool = False
    ):
        path = os.path.expanduser(path) if not isinstance(path, File) else path
        if _check_file_for_mismatches(path, self._from_path(path), values, overwrite):
            return self

        new_secret = copy.deepcopy(self)
        new_secret._values = None
        new_secret.path = path

        token = self.values["token"]
        if isinstance(path, File):
            path.write(token, serialize=False, mode="w")
        else:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            with open(path, "a") as f:
                f.write(token)
            new_secret._add_to_rh_config(path)

        return new_secret

    def _from_path(self, path: Union[str, File]):
        token = None
        if isinstance(path, File):
            if path.exists_in_system():
                token = path.fetch(mode="r").strip("\n")
        elif path and os.path.exists(os.path.expanduser(path)):
            token = Path(os.path.expanduser(path)).read_text().strip("\n")
        if token:
            return {"token": token}
        return {}
