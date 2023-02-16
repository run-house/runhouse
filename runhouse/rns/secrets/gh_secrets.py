import os
from typing import Optional

from runhouse import Secrets


# TODO [DG] untested, test this
class GHSecrets(Secrets):
    PROVIDER_NAME = "gh"
    CREDENTIALS_FILE = os.path.expanduser("~/.config/gh/hosts.yml")

    @classmethod
    def read_secrets(cls, from_env: bool = False, file_path: Optional[str] = None):
        if from_env:
            raise NotImplementedError(
                f"Reading secrets from env is not supported for {cls.PROVIDER_NAME}"
            )
        else:
            creds_file = file_path or cls.default_credentials_path()
            config_data = cls.read_yaml_file(creds_file)
            token = config_data["github.com"]["oauth_token"]

        return {"provider": cls.PROVIDER_NAME, "token": token}

    @classmethod
    def save_secrets(
        cls, secrets: dict, file_path: Optional[str] = None, overwrite: bool = False
    ):
        dest_path = file_path or cls.default_credentials_path()
        config = cls.read_yaml_file(dest_path) if cls.file_exists(dest_path) else {}
        config["github.com"] = {"oauth_token": secrets["token"]}

        if cls.has_secrets_file() and not overwrite:
            cls.check_secrets_for_mismatches(
                secrets_to_save=secrets, file_path=dest_path
            )
            return

        cls.save_to_yaml_file(config, dest_path)
        cls.save_secret_to_config()
