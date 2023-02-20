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

        return {"api_key": api_key}

    @classmethod
    def save_secrets(cls, secrets: dict, overwrite: bool = False):

        dest_path = cls.default_credentials_path()
        cls.check_secrets_for_mismatches(
            secrets_to_save=secrets, secrets_path=dest_path, overwrite=overwrite
        )

        Path(dest_path).parent.mkdir(parents=True, exist_ok=True)

        with open(dest_path, "w") as f:
            f.write(f'api_key = {secrets["api_key"]}\n')

        cls.add_provider_to_rh_config()
