import configparser
import os
from typing import Optional

from runhouse import Secrets


class AzureSecrets(Secrets):
    PROVIDER_NAME = "azure"
    CREDENTIALS_FILE = os.path.expanduser("~/.azure/clouds.config")

    @classmethod
    def read_secrets(cls, from_env: bool = False, file_path: Optional[str] = None):
        if from_env:
            subscription_id = os.getenv("AZURE_SUBSCRIPTION_ID")
            if not subscription_id:
                raise Exception(
                    f"AZURE_SUBSCRIPTION_ID must is not set for {cls.PROVIDER_NAME}"
                )
        else:
            creds_file = file_path or cls.default_credentials_path()
            config = cls.read_config_file(creds_file)
            subscription_id = config["AzureCloud"]["subscription"]

        return {"subscription_id": subscription_id}

    @classmethod
    def save_secrets(cls, secrets: dict, overwrite: bool = False):

        dest_path = cls.default_credentials_path()
        cls.check_secrets_for_mismatches(
            secrets_to_save=secrets, secrets_path=dest_path, overwrite=overwrite
        )

        parser = configparser.ConfigParser()
        section_name = "AzureCloud"
        parser.add_section(section_name)
        parser.set(
            section=section_name,
            option="subscription",
            value=secrets["subscription_id"],
        )

        cls.save_to_config_file(parser, dest_path)
        cls.add_provider_to_rh_config()
