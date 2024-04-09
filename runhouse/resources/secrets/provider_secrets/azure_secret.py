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
    """
    .. note::
            To create an AzureSecret, please use the factory method :func:`provider_secret` with ``provider="azure"``.
    """

    # values format: {"subscription_id": subscription_id}
    _PROVIDER = "azure"
    _DEFAULT_CREDENTIALS_PATH = "~/.azure/clouds.config"
    _DEFAULT_ENV_VARS = {"subscription_id": "AZURE_SUBSCRIPTION_ID"}

    @staticmethod
    def from_config(config: dict, dryrun: bool = False, _resolve_children: bool = True):
        return AzureSecret(**config, dryrun=dryrun)

    def _write_to_file(
        self,
        path: Union[str, File] = None,
        values: Dict = None,
        overwrite: bool = False,
    ):
        new_secret = copy.deepcopy(self)
        if not _check_file_for_mismatches(
            path, self._from_path(path), values, overwrite
        ):
            subscription_id = values["subscription_id"]

            parser = configparser.ConfigParser()
            section_name = "AzureCloud"
            parser.add_section(section_name)
            parser.set(
                section=section_name,
                option="subscription",
                value=subscription_id,
            )

            if isinstance(path, File):
                with io.StringIO() as ss:
                    parser.write(ss)
                    ss.seek(0)
                    data = ss.read()
                path.write(data, serialize=False, mode="w")
            else:
                full_path = os.path.expanduser(path)
                Path(full_path).parent.mkdir(parents=True, exist_ok=True)
                with open(full_path, "w") as f:
                    parser.write(f)
                new_secret._add_to_rh_config(path)

        new_secret._values = None
        new_secret.path = path
        return new_secret

    def _from_path(self, path: Union[str, File]):
        config = configparser.ConfigParser()
        if isinstance(path, File):
            if not path.exists_in_system():
                return {}
            config.read_string(path.fetch(mode="r", deserialize=False))
        elif path and os.path.exists(os.path.expanduser(path)):
            path = os.path.expanduser(path)
            config.read(path)
        if config and "AzureCloud" in config.sections():
            subscription_id = config["AzureCloud"]["subscription"]
            return {"subscription_id": subscription_id}
        return {}
