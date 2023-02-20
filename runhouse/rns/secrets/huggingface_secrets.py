import os
import subprocess
from pathlib import Path
from typing import Optional

from runhouse import Secrets


class HuggingFaceSecrets(Secrets):
    PROVIDER_NAME = "huggingface"
    CREDENTIALS_FILE = os.path.expanduser("~/.huggingface/token")

    @classmethod
    def read_secrets(cls, from_env: bool = False, file_path: Optional[str] = None):
        if from_env:
            raise NotImplementedError(
                f"Reading secrets from env is not supported for {cls.PROVIDER_NAME}"
            )
        else:
            creds_path = file_path or cls.default_credentials_path()
            token = Path(creds_path).read_text()

        return {"token": token}

    @classmethod
    def save_secrets(cls, secrets: dict, overwrite: bool = False):
        # TODO check properly if hf needs to be installed
        try:
            import huggingface_hub
        except ModuleNotFoundError:
            subprocess.run(["pip", "install", "--upgrade", "huggingface-hub"])
            import huggingface_hub

        dest_path = cls.default_credentials_path()
        cls.check_secrets_for_mismatches(
            secrets_to_save=secrets, secrets_path=dest_path, overwrite=overwrite
        )

        huggingface_hub.login(token=secrets["token"])
        cls.add_provider_to_rh_config()
