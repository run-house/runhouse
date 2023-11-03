import configparser
import copy
import io
import os
from pathlib import Path

from typing import Dict, Union

from runhouse.resources.blobs.file import File
from runhouse.resources.secrets.provider_secrets.provider_secret import ProviderSecret
from runhouse.resources.secrets.utils import _check_file_for_mismatches


class AzureSecret(ProviderSecret):
    # values format: {"subscription_id": subscription_id}
    _PROVIDER = "azure"
    _DEFAULT_CREDENTIALS_PATH = "~/.azure/clouds.config"
    _DEFAULT_ENV_VARS = {"subscription_id": "AZURE_SUBSCRIPTION_ID"}

    @staticmethod
    def from_config(config: dict, dryrun: bool = False):
        return AzureSecret(**config, dryrun=dryrun)

    def _write_to_file(
        self,
        path: Union[str, File] = None,
        values: Dict = None,
        overwrite: bool = False,
    ):
        path = os.path.expanduser(path) if not isinstance(path, File) else path
        if _check_file_for_mismatches(path, self._from_path(path), values, overwrite):
            return self

        subscription_id = values["subscription_id"]

        parser = configparser.ConfigParser()
        section_name = "AzureCloud"
        parser.add_section(section_name)
        parser.set(
            section=section_name,
            option="subscription",
            value=subscription_id,
        )

        new_secret = copy.deepcopy(self)
        new_secret._values = None
        new_secret.path = path

        if isinstance(path, File):
            with io.StringIO() as ss:
                parser.write(ss)
                ss.seek(0)
                data = ss.read()
            path.write(data, serialize=False, mode="w")
        else:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w") as f:
                parser.write(f)
            new_secret._add_to_rh_config(path)

        return new_secret

    def _from_path(self, path: Union[str, File]):
        if isinstance(path, File):
            if not path.exists_in_system():
                return {}
            config.read_string(path.fetch(model="r"))
        elif path and os.path.exists(os.path.expanduser(path)):
            path = os.path.expanduser(path)
            config = configparser.ConfigParser()
            config.read(path)
            subscription_id = config["AzureCloud"]["subscription"]
            return {"subscription_id": subscription_id}
        return {}
