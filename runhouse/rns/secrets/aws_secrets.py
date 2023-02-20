import configparser
import os
from typing import Optional

from runhouse import Secrets


class AWSSecrets(Secrets):
    PROVIDER_NAME = "aws"
    CREDENTIALS_FILE = os.path.expanduser("~/.aws/credentials")

    @classmethod
    def read_secrets(cls, from_env: bool = False, file_path: Optional[str] = None):
        if from_env:
            access_key = os.getenv("AWS_ACCESS_KEY_ID")
            secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
            if not access_key or not secret_key:
                raise Exception(
                    f"AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY must be set for {cls.PROVIDER_NAME}"
                )
        else:
            if file_path:
                # Read the credentials from the given file path
                config = cls.read_config_file(file_path)
                section_name = "default"
                access_key = config[section_name]["aws_access_key_id"]
                secret_key = config[section_name]["aws_secret_access_key"]
            else:
                # TODO check if user has boto installed, if not tell them to install with runhouse[aws]
                import boto3

                session = boto3.Session()
                credentials = session.get_credentials()

                # Credentials are refreshable, so accessing your access key / secret key
                # separately can lead to a race condition.
                credentials = credentials.get_frozen_credentials()
                access_key = credentials.access_key
                secret_key = credentials.secret_key

        return {
            "access_key": access_key,
            "secret_key": secret_key,
        }

    @classmethod
    def save_secrets(cls, secrets: dict, overwrite: bool = False):
        dest_path = cls.default_credentials_path()
        cls.check_secrets_for_mismatches(
            secrets_to_save=secrets, secrets_path=dest_path, overwrite=overwrite
        )

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

        cls.save_to_config_file(parser, dest_path)
        cls.add_provider_to_rh_config()
