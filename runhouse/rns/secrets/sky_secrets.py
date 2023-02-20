import os
from pathlib import Path
from typing import Optional, Tuple

import sky

from runhouse import Secrets


class SkySecrets(Secrets):
    PROVIDER_NAME = "sky"
    PRIVATE_KEY_FILE = os.path.expanduser(
        sky.authentication.PRIVATE_SSH_KEY_PATH
    )  # '~/.ssh/sky-key'
    PUBLIC_KEY_FILE = os.path.expanduser(
        sky.authentication.PUBLIC_SSH_KEY_PATH
    )  # '~/.ssh/sky-key.pub'

    @classmethod
    def default_credentials_path(cls) -> Tuple:
        return cls.PRIVATE_KEY_FILE, cls.PUBLIC_KEY_FILE

    @classmethod
    def has_secrets_file(cls) -> bool:
        secret_files: tuple = cls.default_credentials_path()
        if not secret_files:
            return False
        return all(cls.file_exists(s) for s in secret_files)

    @classmethod
    def read_secrets(cls, from_env: bool = False, file_path: Optional[str] = None):
        if from_env:
            raise NotImplementedError(
                f"Reading secrets from env is not supported for {cls.PROVIDER_NAME}"
            )
        else:
            private_key_file = file_path or cls.PRIVATE_KEY_FILE
            public_key_file = file_path or cls.PUBLIC_KEY_FILE

            private_key = Path(private_key_file).read_text()
            public_key = Path(public_key_file).read_text()

        return {
            "ssh_private_key": private_key,
            "ssh_public_key": public_key,
        }

    @classmethod
    def save_secrets(cls, secrets: dict, overwrite: bool = False):
        private_key_path = cls.PRIVATE_KEY_FILE
        if private_key_path.endswith(".pem"):
            public_key_path = private_key_path.rsplit(".", 1)[0] + ".pub"
        else:
            public_key_path = private_key_path + ".pub"

        cls.check_secrets_for_mismatches(
            secrets_to_save=secrets, secrets_path=public_key_path, overwrite=overwrite
        )

        sky.authentication._save_key_pair(
            private_key_path,
            public_key_path,
            secrets["ssh_private_key"],
            secrets["ssh_public_key"],
        )
        # TODO do we need to register the keys with cloud providers? Probably not, sky does this for us later.
        # backend_utils._add_auth_to_cluster_config(sky.clouds.CLOUD_REGISTRY.from_str(self.provider),
        #                                                   Path(yaml_path).expanduser())
        cls.add_provider_to_rh_config()
