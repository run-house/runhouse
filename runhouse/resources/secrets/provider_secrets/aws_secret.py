import configparser
import copy
import os
from pathlib import Path

from typing import Optional

from runhouse.resources.blobs.file import File
from runhouse.resources.secrets.functions import _check_file_for_mismatches
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
        overwrite: bool = False,
    ):
        """Write down the AWS credentials values.

        Args:
            path (str or Path, optional): File to write down the aws secret to. If not provided,
                will use the path associated with the secret object.
        """
        # TODO [secrets] -- add next couple of lines into provider_secret.write()
        new_secret = copy.deepcopy(self)
        if path:
            new_secret.path = path
        path = path or self.path
        path = os.path.expanduser(path)
        values = self.values

        if os.path.exists(path) and _check_file_for_mismatches(
            path, self._from_path(path), values, overwrite
        ):
            return self

        parser = configparser.ConfigParser()
        section_name = "default"
        parser.add_section(section_name)
        parser.set(
            section=section_name,
            option="aws_access_key_id",
            value=values["access_key"],
        )
        parser.set(
            section=section_name,
            option="aws_secret_access_key",
            value=values["secret_key"],
        )

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w+") as f:
            parser.write(f)

        return new_secret

    def _from_path(self, path: Optional[str] = None):
        path = path or self.path
        config = configparser.ConfigParser()
        if isinstance(path, File):
            config.read_string(path.fetch(mode="r"))
        elif path and os.path.exists(os.path.expanduser(path)):
            config.read(os.path.expanduser(path))
        else:
            return {}

        section_name = "default"
        access_key = config[section_name]["aws_access_key_id"]
        secret_key = config[section_name]["aws_secret_access_key"]

        return {
            "access_key": access_key,
            "secret_key": secret_key,
        }
