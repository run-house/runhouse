import os
from pathlib import Path
from typing import Optional

from runhouse import Secrets


class LambdaSecrets(Secrets):
    PROVIDER_NAME = "lambda"
    CREDENTIALS_FILE = os.path.expanduser("~/.lambda_cloud/lambda_keys")

    @classmethod
    def read_secrets(cls, from_env: bool = False, file_path: Optional[str] = None):
        if from_env:
            raise NotImplementedError(
                "Lambda secrets cannot be read from environment variables."
            )
        else:
            from sky.clouds.lambda_cloud import lambda_utils

            client = lambda_utils.LambdaCloudClient()
            api_key = client.api_key
            ssh_key_name = client.ssh_key_name

        return {
            "provider": cls.PROVIDER_NAME,
            "api_key": api_key,
            "ssh_key_name": ssh_key_name,
        }

    @classmethod
    def save_secrets(
        cls, secrets: dict, file_path: Optional[str] = None, overwrite: bool = False
    ):
        dest_path = file_path or cls.default_credentials_path()
        Path(dest_path).parent.mkdir(parents=True, exist_ok=True)

        if cls.has_secrets_file() and not overwrite:
            cls.check_secrets_for_mismatches(
                secrets_to_save=secrets, file_path=dest_path
            )
            return

        with open(dest_path, "w") as f:
            f.write(f'api_key = {secrets["api_key"]}\n')
            f.write(f'ssh_key_name = {secrets["ssh_key_name"]}\n')

        cls.save_secret_to_config()
