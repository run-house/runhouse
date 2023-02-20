import os
from pathlib import Path
from typing import Optional

from runhouse import Secrets


class GitHubSecrets(Secrets):
    PROVIDER_NAME = "github"
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

        return {"token": token}

    @classmethod
    def save_secrets(cls, secrets: dict, overwrite: bool = False):

        dest_path = cls.default_credentials_path()
        cls.check_secrets_for_mismatches(
            secrets_to_save=secrets, secrets_path=dest_path, overwrite=overwrite
        )

        config = cls.read_yaml_file(dest_path) if cls.file_exists(dest_path) else {}
        config["github.com"] = {"oauth_token": secrets["token"]}

        Path(dest_path).parent.mkdir(parents=True, exist_ok=True)

        cls.save_to_yaml_file(config, dest_path)
        cls.add_provider_to_rh_config()
