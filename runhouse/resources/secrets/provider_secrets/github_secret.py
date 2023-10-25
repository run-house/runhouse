import copy
import os
from pathlib import Path

from typing import Optional

import yaml

from runhouse.resources.blobs.file import File
from runhouse.resources.secrets.provider_secrets.provider_secret import ProviderSecret
from runhouse.resources.secrets.utils import _check_file_for_mismatches


class GitHubSecret(ProviderSecret):
    # values format: {"oauth_token": oath_token}
    _DEFAULT_CREDENTIALS_PATH = "~/.config/gh/hosts.yml"
    _PROVIDER = "github"

    @staticmethod
    def from_config(config: dict, dryrun: bool = False):
        return GitHubSecret(**config, dryrun=dryrun)

    def write(
        self,
        path: str = None,
        overwrite: bool = False,
    ):
        """Write down the Github credentials values.

        Args:
            path (str or Path, optional): File to write down the Github secret to. If not provided,
                will use the path associated with the secret object.
        """
        new_secret = copy.deepcopy(self)
        path = path or self.path
        if path:
            new_secret.path = path

        path = os.path.expanduser(path) if not isinstance(path, File) else path
        if _check_file_for_mismatches(
            path, self._from_path(path), self.values, overwrite
        ):
            return self

        values = self.values
        config = {}
        if isinstance(path, File):
            if path.exists_in_system():
                config = path.fetch(deserialize=False, mode="r")
            config["github.com"] = values
            data = yaml.dump(config, default_flow_style=False)
            path.write(data, serialize=False, mode="w")
        else:
            if Path(path).exists():
                with open(path, "r") as stream:
                    config = yaml.safe_load(stream)
            config["github.com"] = values

            Path(path).parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w") as yaml_file:
                yaml.dump(config, yaml_file, default_flow_style=False)

        return new_secret

    def _from_path(self, path: Optional[str] = None):
        path = path or self.path
        config = {}
        if isinstance(path, File):
            if not path.exists_in_system():
                return {}
            config = yaml.safe_load(path.fetch(mode="r"))
        elif path and os.path.exists(os.path.expanduser(path)):
            with open(os.path.expanduser(path), "r") as stream:
                config = yaml.safe_load(stream)
        return config["github.com"] if config else {}
