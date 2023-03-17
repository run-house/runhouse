import logging
import os
from pathlib import Path
from typing import Optional

from runhouse import Secrets

logger = logging.getLogger(__name__)


class SSHSecrets(Secrets):
    PROVIDER_NAME = "ssh"
    CREDENTIALS_FILE = os.path.expanduser("~/.ssh")

    @classmethod
    def read_secrets(cls, from_env: bool = False, file_path: Optional[str] = None):
        if from_env:
            raise NotImplementedError(
                "SSH secrets cannot be read from environment variables."
            )
        else:
            creds_path = Path(file_path or cls.CREDENTIALS_FILE).expanduser()
            config_data = {}
            for f in creds_path.glob("*"):
                # TODO do we need to support pem files?
                if f.suffix != ".pub":
                    continue
                if f.name == "sky-key.pub":
                    # We don't need to store duplicate ssh keys for sky (Sky is already a builtin provider)
                    continue
                if not Path(creds_path / f.stem).exists():
                    logger.warning(
                        f"Private key {f.stem} not found for public key {f.name}, skipping."
                    )
                    continue
                # Grab public key
                config_data[f.name] = Path(creds_path / f).read_text()
                # Grab corresponding private key
                config_data[f.stem] = Path(creds_path / f.stem).read_text()

        return config_data

    @classmethod
    def save_secrets(cls, secrets: dict, overwrite: bool = False):
        dest_path = Path(cls.default_credentials_path()).expanduser()
        cls.check_secrets_for_mismatches(
            secrets_to_save=secrets, secrets_path=str(dest_path), overwrite=overwrite
        )

        dest_path.mkdir(parents=True, exist_ok=True)

        valid_secrets = {}
        for key_name, key in secrets.items():
            if key_name == "provider":
                continue
            key_path = dest_path / key_name
            if key_path.exists():
                logger.warning(f"Key {key_name} already exists, skipping.")
                continue
            key_path.write_text(key)
            key_path.chmod(0o600)
            valid_secrets[key_name] = str(key_path)

        cls.add_provider_to_rh_config(secrets_for_config={cls.PROVIDER_NAME: secrets})
