import copy
import logging
import os
from pathlib import Path

from typing import Any, Dict, Optional, Union

from runhouse.globals import configs, rns_client

from runhouse.resources.blobs.file import File
from runhouse.resources.hardware.cluster import Cluster
from runhouse.resources.secrets.provider_secrets.provider_secret import ProviderSecret

logger = logging.getLogger(__name__)


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
    def from_config(config: dict, dryrun: bool = False):
        # try block if for the case we are trying to load a shared secret.
        try:
            secret_creator = config["owner"]["username"]
            current_user = configs.username
            if secret_creator == current_user:
                new_values = config.pop("values")
                config["values"] = new_values
            else:
                new_values = config.pop("values")
                if "ssh_public_key" in new_values.keys():
                    folder_name = config["name"][1:].replace("/", "_")
                    ssh_path = str(
                        Path("~/.ssh").expanduser() / folder_name / "ssh-key"
                    )
                    private_key_to_write = {
                        "private_key": new_values.pop("private_key", ""),
                        "public_key": new_values.pop("public_key", ""),
                    }
                    SSHSecret._write_to_file(
                        self=SSHSecret, path=ssh_path, values=private_key_to_write
                    )
                    new_values["ssh_private_key"] = ssh_path
                config["values"] = new_values
            return SSHSecret(**config, dryrun=dryrun)
        except KeyError:
            return SSHSecret(**config, dryrun=dryrun)

    def save(
        self, name: str = None, save_values: bool = True, headers: Optional[Dict] = None
    ):

        if name:
            self.name = name
        elif not self.name:
            self.name = f"ssh-{self.key}"
        return super().save(
            save_values=save_values, headers=headers or rns_client.request_headers()
        )

    def _write_to_file(
        self, path: Union[str, File], values: Dict = None, overwrite: bool = False
    ):
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
        priv_key_path.write_text(values["private_key"])
        priv_key_path.chmod(0o600)
        pub_key_path.write_text(values["public_key"])
        pub_key_path.chmod(0o600)

        new_secret = copy.deepcopy(self)
        new_secret._values = None
        new_secret.path = path
        try:
            new_secret._add_to_rh_config(val=path)
        except TypeError:
            pass

        return new_secret

    def _from_path(self, path: Union[str, File], public_path=None):
        if path == self._DEFAULT_CREDENTIALS_PATH:
            path = f"{self._DEFAULT_CREDENTIALS_PATH}/{self.key}"

        if isinstance(path, File):
            from runhouse.resources.blobs.file import file

            priv_key = path.fetch(mode="r", deserialize=False)
            pub_key_file = file(path=f"{path.path}.pub", system=path.system)
            pub_key = pub_key_file.fetch(mode="r", deserialize=False)
        else:
            pub_key_path = (
                os.path.expanduser(f"{path}.pub")
                if not public_path
                else os.path.expanduser(f"{public_path}.pub")
                if ".pub" not in public_path
                else public_path
            )
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
            system.run([f"chmod 600 {path}"])
        else:
            system.call(key, "_write_to_file", path=path, values=values)
            remote_priv_file = file(path=path, system=system)
        return remote_priv_file

    @classmethod
    def setup_ssh_creds(cls, ssh_creds: Union[Dict, str], resource_name: str):
        """
        this method creates an SSHSecret instance based o n the passed values. If the passed values are paths to private
         and/or public keys, this method extracts the content of the files saved in those files, in order for them to
         be saved in den. (Currently if we just pass a path/to/ssh/key to SSHSecret constructor, the content of the file
         will not be saved to Vault. We need to pass the content itself.
         Args:
            ssh_creds (Dict or str): the ssh credentials passed by the user, dict.
            resource_name (str): the name of the resource that the ssh secret is accosted to.
         Returns:
            SSHSecret: the values of it equal to ssh_creds.
        """
        import runhouse as rh

        if isinstance(ssh_creds, str):
            return cls.from_name(name=ssh_creds)

        creds_keys = list(ssh_creds.keys())

        if len(creds_keys) == 1 and "ssh_private_key" in creds_keys:
            if Path(ssh_creds["ssh_private_key"]).expanduser().exists():
                values = cls._from_path(self=cls, path=ssh_creds["private_key"])
                values["ssh_private_key"] = ssh_creds["private_key"]
            else:
                # case where the user decides to pass the private key as text and not as path.
                raise ValueError(
                    "SSH creds require both private and public key, but only private key was provided"
                )
        elif "ssh_private_key" in creds_keys and "ssh_public key" in creds_keys:
            private_key, public_key = (
                ssh_creds["ssh_private_key"],
                ssh_creds["ssh_public_key"],
            )
            private_key_path, public_key_path = (
                Path(private_key).expanduser(),
                Path(public_key).expanduser(),
            )
            if private_key_path.exists() and public_key_path.exists():
                if private_key_path.parent == public_key_path.parent:
                    values = cls._from_path(self=cls, path=private_key)
                else:
                    values = cls._from_path(
                        self=cls, path=private_key, public_path=public_key
                    )
                values["ssh_private_key"], values["ssh_public_key"] = (
                    private_key_path,
                    public_key_path,
                )
            else:
                values = {"private_key": private_key, "public_key": public_key}
        elif "ssh_private_key" in creds_keys and "ssh_user" in creds_keys:
            private_key, username = ssh_creds["ssh_private_key"], ssh_creds["ssh_user"]
            if Path(private_key).expanduser().exists():
                private_key = cls._from_path(self=cls, path=private_key).get(
                    "private_key"
                )
            if Path(username).expanduser().exists():
                username = super()._from_path(username)
            values = {
                "private_key": private_key,
                "ssh_user": username,
                "ssh_private_key": ssh_creds["ssh_private_key"],
            }
        elif "ssh_user" in creds_keys and "password" in creds_keys:
            password, username = ssh_creds["password"], ssh_creds["ssh_user"]
            if Path(password).expanduser().exists():
                password = super()._from_path(password)
            if Path(username).expanduser().exists():
                username = super()._from_path(username)
            values = {"password": password, "ssh_user": username}
        else:
            values = {}
            for k in creds_keys:
                v = ssh_creds[k]
                if Path(v).exists():
                    v = super()._from_path(v)
                values.update({k: v})
        values_to_add = {k: ssh_creds[k] for k in ssh_creds if k not in values.keys()}
        values.update(values_to_add)
        new_secret = rh.secret(provider="ssh", values=values).save(
            name=f"{resource_name}-ssh-secret"
        )
        return new_secret
