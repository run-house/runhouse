import copy
import os
from pathlib import Path

from typing import Dict

from runhouse.resources.secrets.provider_secrets.provider_secret import ProviderSecret
from runhouse.resources.secrets.utils import _check_file_for_mismatches
from runhouse.utils import create_local_dir


class HuggingFaceSecret(ProviderSecret):
    """
    .. note::
            To create a HuggingFaceSecret, please use the factory method :func:`provider_secret` with
            ``provider="huggingface"``.
    """

    # values format: {"token": hf_token}
    _PROVIDER = "huggingface"
    _DEFAULT_CREDENTIALS_PATH = "~/.cache/huggingface/token"
    _DEFAULT_ENV_VARS = {"token": "HF_TOKEN"}

    @staticmethod
    def from_config(config: dict, dryrun: bool = False, _resolve_children: bool = True):
        return HuggingFaceSecret(**config, dryrun=dryrun)

    def _write_to_file(self, path: str, values: Dict = None, overwrite: bool = False):
        new_secret = copy.deepcopy(self)
        if not _check_file_for_mismatches(
            path, self._from_path(path), values, overwrite
        ):
            token = values["token"]
            full_path = create_local_dir(path)
            with open(full_path, "a") as f:
                f.write(token)
            new_secret._add_to_rh_config(path)

        new_secret._values = None
        new_secret.path = path
        return new_secret

    def _from_path(self, path: str = None):
        token = None
        if path and os.path.exists(os.path.expanduser(path)):
            token = Path(os.path.expanduser(path)).read_text().strip("\n")
        if token:
            return {"token": token}
        return {}
