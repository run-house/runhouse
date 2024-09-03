import configparser
import copy
import os

from typing import Dict

from runhouse.resources.secrets.provider_secrets.provider_secret import ProviderSecret
from runhouse.resources.secrets.utils import _check_file_for_mismatches
from runhouse.utils import create_local_dir


class AWSSecret(ProviderSecret):
    """
    .. note::
            To create an AWSSecret, please use the factory method :func:`provider_secret` with ``provider="aws"``.
    """

    _PROVIDER = "aws"
    _DEFAULT_CREDENTIALS_PATH = "~/.aws/credentials"
    _DEFAULT_ENV_VARS = {
        "access_key": "AWS_ACCESS_KEY_ID",
        "secret_key": "AWS_SECRET_ACCESS_KEY",
    }

    @staticmethod
    def from_config(config: dict, dryrun: bool = False, _resolve_children: bool = True):
        return AWSSecret(**config, dryrun=dryrun)

    def _write_to_file(self, path: str, values: Dict, overwrite: bool = False):
        new_secret = copy.deepcopy(self)

        if not _check_file_for_mismatches(
            path, self._from_path(path), values, overwrite
        ):

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

            full_path = create_local_dir(path)
            with open(full_path, "w+") as f:
                parser.write(f)
            new_secret._add_to_rh_config(path)

        new_secret._values = None
        new_secret.path = path
        return new_secret

    def _from_path(self, path: str):
        config = configparser.ConfigParser()
        if path and os.path.exists(os.path.expanduser(path)):
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
