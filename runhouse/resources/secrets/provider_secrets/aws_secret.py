import configparser
import copy
import os
from pathlib import Path

from typing import Optional

from runhouse.resources.secrets.provider_secrets.provider_secret import ProviderSecret


class AWSSecret(ProviderSecret):
    _DEFAULT_CREDENTIALS_PATH = "~/.aws/credentials"
    _PROVIDER = "aws"
    _ENV_VARS = {
        "access_key": "AWS_ACCESS_KEY_ID",
        "secret_key": "AWS_SECRET_ACCESS_KEY",
    }

    @staticmethod
    def from_config(config: dict, dryrun: bool = False):
        return AWSSecret(**config, dryrun=dryrun)

    def write(
        self,
        path: str = None,
    ):
        """Write down the AWS secrets.

        Args:
            path (str or Path, optional): File to write down the aws secrets to. If not provided,
                will use the path associated with the secrets object.
        """
        new_secret = copy.deepcopy(self)
        if path:
            new_secret.path = path

        path = path or self.path
        path = os.path.expanduser(path)
        secrets = self.secrets

        parser = configparser.ConfigParser()
        section_name = "default"
        parser.add_section(section_name)
        parser.set(
            section=section_name,
            option="aws_access_key_id",
            value=secrets["access_key"],
        )
        parser.set(
            section=section_name,
            option="aws_secret_access_key",
            value=secrets["secret_key"],
        )

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w+") as f:
            parser.write(f)

        return new_secret

    def _from_path(self, path: Optional[str] = None):
        path = path or self.path
        if path and os.path.exists(os.path.expanduser(path)):
            config = configparser.ConfigParser()
            config.read(os.path.expanduser(path))

            section_name = "default"
            access_key = config[section_name]["aws_access_key_id"]
            secret_key = config[section_name]["aws_secret_access_key"]

            return {
                "access_key": access_key,
                "secret_key": secret_key,
            }
        return {}
