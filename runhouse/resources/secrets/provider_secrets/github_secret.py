import copy
import os
from pathlib import Path

from typing import Dict, Union

import yaml

from runhouse.resources.blobs.file import File
from runhouse.resources.secrets.provider_secrets.provider_secret import ProviderSecret
from runhouse.resources.secrets.utils import _check_file_for_mismatches


class GitHubSecret(ProviderSecret):
    """
    .. note::
            To create a GitHubSecret, please use the factory method :func:`provider_secret` with ``provider="github"``.
    """

    # values format: {"oauth_token": oath_token}
    _PROVIDER = "github"
    _DEFAULT_CREDENTIALS_PATH = "~/.config/gh/hosts.yml"

    @staticmethod
    def from_config(config: dict, dryrun: bool = False, _resolve_children: bool = True):
        return GitHubSecret(**config, dryrun=dryrun)

    def _write_to_file(
        self, path: Union[str, File], values: Dict = None, overwrite: bool = False
    ):
        new_secret = copy.deepcopy(self)
        if not _check_file_for_mismatches(
            path, self._from_path(path), values, overwrite
        ):
            config = {}
            if isinstance(path, File):
                if path.exists_in_system():
                    config = path.fetch(deserialize=False, mode="r")
                config["github.com"] = values
                data = yaml.dump(config, default_flow_style=False)
                path.write(data, serialize=False, mode="w")
            else:
                full_path = os.path.expanduser(path)
                if Path(full_path).exists():
                    with open(full_path, "r") as stream:
                        config = yaml.safe_load(stream)
                config["github.com"] = values

                Path(full_path).parent.mkdir(parents=True, exist_ok=True)
                with open(full_path, "w") as yaml_file:
                    yaml.dump(config, yaml_file, default_flow_style=False)
                new_secret._add_to_rh_config(path)

        new_secret._values = None
        new_secret.path = path
        return new_secret

    def _from_path(self, path: Union[str, File]):
        config = {}
        if isinstance(path, File):
            if not path.exists_in_system():
                return {}
            config = yaml.safe_load(path.fetch(mode="r", deserialize=False))
        elif path and os.path.exists(os.path.expanduser(path)):
            with open(os.path.expanduser(path), "r") as stream:
                config = yaml.safe_load(stream)
        return config["github.com"] if config else {}
