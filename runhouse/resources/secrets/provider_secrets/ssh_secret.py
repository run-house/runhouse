import copy
import logging
import os
from pathlib import Path

from typing import Any, Dict, Optional, Union

from runhouse.resources.blobs.file import File
from runhouse.resources.hardware.cluster import Cluster
from runhouse.resources.secrets.provider_secrets.provider_secret import ProviderSecret
from runhouse.rns import rns_client

logger = logging.getLogger(__name__)


class SSHSecret(ProviderSecret):
    _DEFAULT_CREDENTIALS_PATH = "~/.ssh"
    _PROVIDER = "ssh"
    _DEFAULT_KEY = "id_rsa"
    _ENV_VARS = {}

    def __init__(
        self,
        name: Optional[str] = None,
        provider: Optional[str] = None,
        values: Dict = {},
        path: str = None,
        dryrun: bool = True,
        **kwargs,
    ):
        self.key = os.path.basename(path) if path else (name or self._DEFAULT_KEY)
        super().__init__(
            name=name, provider=provider, values=values, path=path, dryrun=dryrun
        )

    def from_config(config: dict, dryrun: bool = False):
        return SSHSecret(**config, dryrun=dryrun)

    def save(self, headers: str = rns_client.request_headers):
        if not self.name:
            self.name = f"ssh-{self.key}"
        super().save(headers=headers)

    def _write_to_file(
        self, path: Union[str, File], values: Dict = None, overwrite: bool = False
    ):
        priv_key_path = path

        priv_key_path = Path(os.path.expanduser(priv_key_path))
        pub_key_path = Path(f"{os.path.expanduser(priv_key_path)}.pub")
        values = self.values

        if priv_key_path.exists() or pub_key_path.exists():
            if values == self._from_path(path):
                logger.info(f"Secrets already exist in {path}. Skipping.")
                return self
            logger.warning(
                f"SSH Secrets for {self.key} already exist in {path}. "
                "Automatically overriding SSH keys is not supported by Runhouse. "
                "Please manually edit these files."
            )
            return self

        priv_key_path.parent.mkdir(parents=True, exist_ok=True)
        priv_key_path.write_text(values["private_key"])
        priv_key_path.chmod(0o600)
        pub_key_path.write_text(values["public_key"])
        pub_key_path.chmod(0o600)

        new_secret = copy.deepcopy(self)
        new_secret._values = None
        new_secret.path = path
        new_secret._add_to_rh_config(path)

        return new_secret

    def _from_path(self, path: Union[str, File]):
        if isinstance(path, File):
            from runhouse.resources.blobs.file import file

            priv_key = path.fetch(mode="r")
            pub_key_file = file(path=f"{priv_key.path}.pub", system=priv_key.system)
            pub_key = pub_key_file.fetch(mode="r")
        else:
            pub_key_path = os.path.expanduser(f"{path}.pub")
            priv_key_path = os.path.expanduser(path)

            if not (os.path.exists(pub_key_path) and os.path.exists(priv_key_path)):
                return {}

            pub_key = Path(pub_key_path).read_text()
            priv_key = Path(priv_key_path).read_text()

        return {"public_key": pub_key, "private_key": priv_key}

    def _file_to(
        self,
        key: str,
        system: Union[str, Cluster],
        path: Union[str, Path] = None,
        values: Any = None,
    ):
        from runhouse.resources.blobs.file import file

        if self.path:
            pub_key_path = (
                f"{path.path}.pub" if isinstance(path, File) else f"{path}.pub"
            )
            remote_priv_file = file(path=self.path).to(system, path=path)
            file(path=pub_key_path).to(system, path=pub_key_path)
        else:
            system.call(key, "_write_to_file", path=path, values=values)
            remote_priv_file = file(path=path, system=system)
        return remote_priv_file
