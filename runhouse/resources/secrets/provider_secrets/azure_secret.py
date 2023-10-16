import configparser
import copy
import os
from pathlib import Path

from typing import Optional

from runhouse.resources.blobs.file import File
from runhouse.resources.secrets.functions import _check_file_for_mismatches
from runhouse.resources.secrets.provider_secrets.provider_secret import ProviderSecret


class AzureSecret(ProviderSecret):
    # values format: {"subscription_id": subscription_id}

    _DEFAULT_CREDENTIALS_PATH = "~/.azure/clouds.config"
    _PROVIDER = "azure"
    _ENV_VARS = {"subscription_id": "AZURE_SUBSCRIPTION_ID"}

    @staticmethod
    def from_config(config: dict, dryrun: bool = False):
        return AzureSecret(**config, dryrun=dryrun)

    def write(
        self,
        path: str = None,
        overwrite: bool = False,
    ):
        """Write down the Azure credentials values.

        Args:
            path (str or Path, optional): File to write down the Azure secret to. If not provided,
                will use the path associated with the secret object.
        """
        new_secret = copy.deepcopy(self)
        if path:
            new_secret.path = path
        path = path or self.path
        path = os.path.expanduser(path)
        if os.path.exists(path) and _check_file_for_mismatches(
            path, self._from_path(path), self.values, overwrite
        ):
            return self

        subscription_id = self.values["subscription_id"]

        parser = configparser.ConfigParser()
        section_name = "AzureCloud"
        parser.add_section(section_name)
        parser.set(
            section=section_name,
            option="subscription",
            value=subscription_id,
        )
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            parser.write(f)

        return self

    def _from_path(self, path: Optional[str] = None):
        path = path or self.path

        if isinstance(path, File):
            config.read_string(path.fetch(model="r"))
        elif path and os.path.exists(os.path.expanduser(path)):
            path = os.path.expanduser(path)
            config = configparser.ConfigParser()
            config.read(path)
            subscription_id = config["AzureCloud"]["subscription"]
            return {"subscription_id": subscription_id}
        return {}
