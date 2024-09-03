import copy
import os
from pathlib import Path

from typing import Any, Dict, Optional, Union

from runhouse.globals import rns_client
from runhouse.logger import get_logger
from runhouse.resources.hardware.cluster import Cluster
from runhouse.resources.secrets.provider_secrets.provider_secret import ProviderSecret

logger = get_logger(__name__)


class SSHSecret(ProviderSecret):
    """
    .. note::
            To create a SSHSecret, please use the factory method :func:`provider_secret` with ``provider="ssh"``.
    """

    _DEFAULT_CREDENTIALS_PATH = "~/.ssh"
    _PROVIDER = "ssh"
    _DEFAULT_KEY = "id_rsa"

    def __init__(
        self,
        name: Optional[str] = None,
        provider: Optional[str] = None,
        values: Dict = {},
        path: str = None,
        key: str = None,
        dryrun: bool = True,
        **kwargs,
    ):
        self.key = (
            key or os.path.basename(path) if path else (name or self._DEFAULT_KEY)
        )
        super().__init__(
            name=name, provider=provider, values=values, path=path, dryrun=dryrun
        )
        if self.path == self._DEFAULT_CREDENTIALS_PATH:
            self.path = str(Path(self._DEFAULT_CREDENTIALS_PATH) / self.key)

    @staticmethod
    def from_config(config: dict, dryrun: bool = False, _resolve_children: bool = True):
        # try block if for the case we are trying to load a shared secret.
        return SSHSecret(**config, dryrun=dryrun)

    def save(
        self,
        name: str = None,
        save_values: bool = True,
        headers: Optional[Dict] = None,
        folder: str = None,
    ):
        if name:
            self.name = name
        elif not self.name:
            self.name = f"ssh-{self.key}"
        return super().save(
            save_values=save_values,
            headers=headers or rns_client.request_headers(),
            folder=folder,
        )

    def _write_to_file(self, path: str, values: Dict = None, overwrite: bool = False):
        priv_key_path = path

        priv_key_path = Path(os.path.expanduser(priv_key_path))
        pub_key_path = Path(f"{os.path.expanduser(priv_key_path)}.pub")

        if priv_key_path.exists() and pub_key_path.exists():
            if values == self._from_path(path=path):
                logger.info(f"Secrets already exist in {path}. Skipping.")
                self.path = path
                return self
            logger.warning(
                f"SSH Secrets for {self.name or self.key} already exist in {path}. "
                "Automatically overriding SSH keys is not supported by Runhouse. "
                "Please manually edit these files."
            )
            self.path = path
            return self

        priv_key_path.parent.mkdir(parents=True, exist_ok=True)
        private_key = values.get("private_key")
        if private_key is not None:
            priv_key_path.write_text(private_key)
            priv_key_path.chmod(0o600)
        public_key = values.get("public_key")
        if public_key is not None:
            pub_key_path.write_text(public_key)
            pub_key_path.chmod(0o600)

        new_secret = copy.deepcopy(self)
        new_secret._values = None
        new_secret.path = path
        try:
            new_secret._add_to_rh_config(val=path)
        except TypeError:
            pass

        return new_secret

    def _from_path(self, path: str):
        if path == self._DEFAULT_CREDENTIALS_PATH:
            path = f"{self._DEFAULT_CREDENTIALS_PATH}/{self.key}"

        return self.extract_secrets_from_path(path)

    @staticmethod
    def extract_secrets_from_path(path: str) -> Dict:
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
        if self.path:
            remote_priv_file = self.path
            # pub_key_path = f"{path}.pub"
            system.call(key, "_write_to_file", path=remote_priv_file, values=values)
            system.run([f"chmod 600 {path}"])
        else:
            system.call(key, "_write_to_file", path=path, values=values)
            remote_priv_file = path
        return remote_priv_file
