import copy
import os
from pathlib import Path

from typing import Optional

from runhouse.resources.blobs.file import File
from runhouse.resources.secrets.functions import _check_file_for_mismatches
from runhouse.resources.secrets.provider_secrets.provider_secret import ProviderSecret


class HuggingFaceSecret(ProviderSecret):
    # values format: {"token": hf_token}
    _DEFAULT_CREDENTIALS_PATH = "~/.cache/huggingface/token"
    _PROVIDER = "huggingface"

    @staticmethod
    def from_config(config: dict, dryrun: bool = False):
        return HuggingFaceSecret(**config, dryrun=dryrun)

    def write(self, path: str = None, overwrite: bool = False):
        """Write down the HuggingFace credentials values.

        Args:
            path (str or Path, optional): File to write down the HuggingFace secret to. If not provided,
                will use the path associated with the secret object.
        """
        new_secret = copy.deepcopy(self)
        if path:
            new_secret.path = path
        path = path or self.path
        path = os.path.expanduser(path)
        if os.path.exists(path) and _check_file_for_mismatches(
            path, self._from_path(path), self.values, overwrite
        ):
            return self

        token = self.values["token"]
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a") as f:
            f.write(token)

        return new_secret

    def _from_path(self, path: Optional[str] = None):
        path = path or self.path
        token = None
        if isinstance(path, File):
            token = path.fetch(mode="r").strip("\n")
        elif path and os.path.exists(os.path.expanduser(path)):
            token = Path(os.path.expanduser(path)).read_text().strip("\n")
        if token:
            return {"token": token}
        return {}
