import os
import sys
from typing import Optional

from runhouse import Secrets


class GCPSecrets(Secrets):
    PROVIDER_NAME = "gcp"
    CREDENTIALS_FILE = os.path.expanduser(
        "~/.config/gcloud/application_default_credentials.json"
    )

    @classmethod
    def read_secrets(cls, from_env: bool = False, file_path: Optional[str] = None):
        if from_env:
            client_id = os.getenv("CLIENT_ID")
            client_secret = os.getenv("CLIENT_SECRET")
            if not client_id or not client_secret:
                raise Exception(
                    f"CLIENT_ID and CLIENT_SECRET must be set for {cls.PROVIDER_NAME}"
                )
        else:
            creds_file = file_path or cls.default_credentials_path()
            config_data = cls.read_json_file(creds_file)
            client_id = config_data["client_id"]
            client_secret = config_data["client_secret"]

        return {
            "client_id": client_id,
            "client_secret": client_secret,
        }

    @classmethod
    def save_secrets(cls, secrets: dict, overwrite: bool = False):

        dest_path = cls.default_credentials_path()
        cls.check_secrets_for_mismatches(
            secrets_to_save=secrets, secrets_path=dest_path, overwrite=overwrite
        )

        config = cls.read_json_file(dest_path) if cls.file_exists(dest_path) else {}
        config["client_id"] = secrets["client_id"]
        config["client_secret"] = secrets["client_secret"]

        # We need to do extra stuff if we're in a colab
        if "google.colab" in sys.modules:
            from rich.console import Console

            console = Console()
            console.print(
                "Please do the following to complete gcp secrets setup:",
                style="bold yellow",
            )
            console.print("!gcloud init", style="bold yellow")
            console.print("!gcloud auth application-default login", style="bold yellow")
            console.print(
                "!cp -r /content/.config/* ~/.config/gcloud", style="bold yellow"
            )

        cls.save_to_json_file(config, dest_path)
        cls.add_provider_to_rh_config()
