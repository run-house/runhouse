import json
import logging
import os
import pkgutil
import shutil
from pathlib import Path
from typing import Optional

import requests

from runhouse.rns.api_utils.resource_access import ResourceAccess
from runhouse.rns.api_utils.utils import (
    generate_uuid,
    load_resp_content,
    read_resp_data,
    remove_null_values_from_dict,
)

logger = logging.getLogger(__name__)


class RNSClient:
    """Manage a particular resource with the runhouse database"""

    CORE_RNS_FIELDS = ["name", "resource_type", "folder", "users", "groups"]
    # RH_BUILTINS_FOLDER = '/builtins'
    DEFAULT_FS = "file"

    def __init__(self, configs) -> None:
        self._configs = configs
        self._prev_folders = []

        self.rh_directory = str(Path(self.locate_working_dir()) / "rh")
        self.rh_builtins_directory = str(
            Path(pkgutil.get_loader("runhouse").path).parent / "builtins"
        )

        # TODO allow users to register other base folders
        # Register all the directories in rh folder as rns base folders
        rns_base_folders = (
            [
                str(subdir)
                for subdir in Path(self.rh_directory).iterdir()
                if subdir.is_dir()
            ]
            if Path(self.rh_directory).exists()
            else []
        )
        rns_base_folders.append(
            str(Path(pkgutil.get_loader("runhouse").path).parent / "builtins")
        )
        self._index_base_folders(rns_base_folders)
        self._current_folder = None

        self.refresh_defaults()

    # TODO [DG] move the below into Defaults() so they never need to be refreshed?
    def refresh_defaults(self):
        use_local_configs = (
            ["local"] if self._configs.get("use_local_configs", True) else []
        )
        use_rns = (
            ["rns"]
            if self._configs.get("use_rns", self._configs.get("token", False))
            else []
        )

        self.save_to = use_local_configs + use_rns
        self.load_from = use_local_configs + use_rns

        if self.token is None:
            self.save_to.pop(
                self.save_to.index("rns")
            ) if "rns" in self.save_to else self.save_to
            self.load_from.pop(
                self.load_from.index("rns")
            ) if "rns" in self.load_from else self.load_from
            logger.info(
                "No auth token provided, so not using RNS API to save and load configs"
            )

    @classmethod
    def find_parent_with_file(cls, dir_path, file):
        if Path(dir_path) == Path.home() or dir_path == Path("/"):
            return None
        if Path(dir_path, file).exists():
            return str(dir_path)
        else:
            return cls.find_parent_with_file(Path(dir_path).parent, file)

    @classmethod
    def locate_working_dir(cls, cwd=os.getcwd()):
        # Search for working_dir by looking up directory tree, in the following order:
        # 1. Upward directory with rh/ subdirectory
        # 2. Root git directory
        # 3. Upward directory with requirements.txt
        # 4. User's cwd

        for search_target in [
            "rh",
            ".git",
            "requirements.txt",
            "setup.py",
            "pyproject.toml",
        ]:
            dir_with_target = cls.find_parent_with_file(cwd, search_target)
            if dir_with_target is not None:
                return dir_with_target
        else:
            return cwd

    @property
    def default_folder(self):
        folder = self._configs.get("default_folder", None)
        if folder in [None, "~"] and self._configs.get("username"):
            folder = "/" + self._configs.get("username")
            self._configs.set("default_folder", folder)
        return folder

    @property
    def current_folder(self):
        if not self._current_folder:
            self._current_folder = self.default_folder
        return self._current_folder

    @current_folder.setter
    def current_folder(self, value):
        self._current_folder = value

    @property
    def token(self):
        return self._configs.get("token", None)

    @property
    def api_server_url(self):
        return self._configs.get("api_server_url", None)

    def _index_base_folders(self, lst):
        self.rns_base_folders = {}
        for folder in lst:
            config = self._load_config_from_local(path=folder)
            rns_path = str(Path(self.default_folder) / Path(folder).name)
            if config:
                rns_path = config.get("rns_address")
            self.rns_base_folders[rns_path] = folder

    @staticmethod
    def resource_uri(name):
        """URI used when querying the RNS server"""
        from runhouse.rns.top_level_rns_fns import resolve_rns_path

        rns_address = resolve_rns_path(name)
        return RNSClient.format_rns_address(rns_address)

    @staticmethod
    def format_rns_address(rns_address: str):
        if rns_address.startswith("/"):
            rns_address = rns_address[1:]
        return rns_address.replace("/", ":")

    @staticmethod
    def local_to_remote_address(rns_address):
        return rns_address.replace("~", "@")

    def remote_to_local_address(self, rns_address):
        return rns_address.replace(self.default_folder, "~")

    @property
    def request_headers(self):
        return self._configs.request_headers

    def resource_request_payload(self, payload) -> dict:
        payload = remove_null_values_from_dict(payload)
        data = {}
        for k, v in payload.copy().items():
            if k not in self.CORE_RNS_FIELDS:
                data[k] = v
                # if adding to data field remove as standalone field
                del payload[k]
        payload["data"] = data
        return payload

    def grant_resource_access(
        self,
        rns_address: str,
        user_emails: list,
        access_type: ResourceAccess,
        notify_users: bool,
    ):
        resource_uri = self.resource_uri(rns_address)
        headers = self.request_headers
        access_payload = {
            "users": user_emails,
            "access_type": access_type,
            "notify_users": notify_users,
        }
        uri = "resource/" + resource_uri
        resp = requests.put(
            f"{self.api_server_url}/{uri}/users/access",
            data=json.dumps(access_payload),
            headers=headers,
        )
        if resp.status_code != 200:
            raise Exception(f"Failed to grant access: {load_resp_content(resp)}")

        resp_data: dict = read_resp_data(resp)
        added_users: dict = resp_data.get("added_users", {})
        new_users: dict = resp_data.get("new_users", {})

        return added_users, new_users

    def load_config(
        self,
        name,
    ) -> dict:
        if not name:
            return {}

        rns_address = self.resolve_rns_path(name)

        if rns_address[0] in ["~", "^"]:
            config = self._load_config_from_local(rns_address)
            if config:
                return config

        if rns_address.startswith("/"):
            resource_uri = self.resource_uri(name)
            logger.info(f"Attempting to load config for {rns_address} from RNS.")
            uri = "resource/" + resource_uri
            resp = requests.get(
                f"{self.api_server_url}/{uri}", headers=self.request_headers
            )
            if resp.status_code != 200:
                logger.info(f"No config found in RNS: {load_resp_content(resp)}")
                # No config found, so return empty config
                return {}

            config: dict = read_resp_data(resp)
            if config.get("data", None):
                config.update(config["data"])
                del config["data"]
            return config
        return {}

    def _load_config_from_local(self, rns_address=None, path=None) -> Optional[dict]:
        """Load config from local file"""
        # TODO should we handle remote filessytems, or throw an error if system != 'file'?
        if not path:
            path = self.locate(rns_address, resolve_path=False)
            if not path:
                return None
        config_path = Path(path) / "config.json"
        if not config_path.exists():
            return None

        logger.info(f"Loading config from local file {config_path}")
        with open(config_path, "r") as f:
            try:
                config = json.load(f)
            except json.decoder.JSONDecodeError as e:
                logger.error(f"Error loading config from {config_path}: {e}")
                return None
        if rns_address:
            config["name"] = rns_address
        return config

    def get_rns_address_for_local_path(self, local_path):
        """Get RNS address for local path"""
        try:
            rel_path = str(Path(local_path).relative_to(self.rh_directory))
            return "~/" + rel_path
        except ValueError:
            return None

    def save_config(self, resource, overwrite: bool = True):
        """Register the resource, saving it to local config folder and/or RNS config store. Uses the resource's
        `self.config_for_rns` to generate the dict to save."""
        rns_address = resource.rns_address
        config = resource.config_for_rns

        if not overwrite and self.exists(rns_address):
            raise ValueError(
                f"Resource {rns_address} already exists and overwrite is False."
            )

        config["name"] = rns_address

        if rns_address is None:
            return

        if rns_address[0] in ["~", "^"]:
            self._save_config_to_local(config, rns_address)

        if rns_address.startswith("/"):
            self._save_config_in_rns(config, rns_address)

    def _save_config_to_local(self, config: dict, rns_address: str):
        if not rns_address:
            raise ValueError("Cannot save resource without rns address or path.")
        resource_dir = Path(self.locate(rns_address, resolve_path=False))
        resource_dir.mkdir(parents=True, exist_ok=True)
        config_path = resource_dir / "config.json"
        with open(config_path, "w") as f:
            json.dump(config, f, indent=4)
        logger.info(f"Saving config for {rns_address} to: {config_path}")

    def _save_config_in_rns(self, config, resource_name):
        """Update or create resource config in database"""
        logger.info(f"Saving config to RNS: {config}")

        resource_uri = self.resource_uri(resource_name)
        uri = f"resource/{resource_uri}"

        payload = self.resource_request_payload(config)
        headers = self.request_headers
        resp = requests.put(
            f"{self.api_server_url}/{uri}", data=json.dumps(payload), headers=headers
        )
        if resp.status_code == 200:
            logger.info(f"Config updated in RNS for Runhouse URI <{uri}>")
        elif resp.status_code == 422:  # No changes made to existing Resource
            logger.info(f"Config for {uri} has not changed, nothing to update")
        elif resp.status_code == 404:  # Resource not found
            logger.info(f"Saving new resource in RNS for Runhouse URI <{uri}>")
            # Resource does not yet exist, in which case we need to create from scratch
            resp = requests.post(
                f"{self.api_server_url}/resource",
                data=json.dumps(payload),
                headers=headers,
            )
            if resp.status_code != 200:
                raise Exception(
                    f"Failed to create new resource in RNS: {load_resp_content(resp)}"
                )
        else:
            raise Exception(
                f"Failed to save resource <{uri}> in RNS: {load_resp_content(resp)}"
            )

    def delete_configs(
        self,
        resource,
    ):
        rns_address = (
            resource.rns_address
            if hasattr(resource, "rns_address")
            else self.resolve_rns_path(resource)
        )
        if rns_address is None:
            logger.warning("No rns address exists for resource")
            return

        if rns_address[0] in ["~", "^"]:
            path = self.locate(rns_address, resolve_path=False)
            if path and Path(path).exists():
                shutil.rmtree(path)
            else:
                logger.info(
                    f"Cannot delete resource {rns_address}, could not find the local config."
                )

        if rns_address.startswith("/"):
            resource_uri = self.resource_uri(rns_address)
            uri = "resource/" + resource_uri
            resp = requests.delete(
                f"{self.api_server_url}/{uri}", headers=self.request_headers
            )
            if resp.status_code != 200:
                logger.error(f"Failed to delete_configs <{uri}>")
            else:
                logger.info(f"Successfully deleted <{uri}>")

    def resolve_rns_data_resource_name(self, name: str):
        """If no name is explicitly provided for the data resource, we need to create one based on the relevant
        rns path. If name is None, return a hex uuid.
        For example: my_blob -> jlewitt1/my_blob"
        """
        if name is None:
            return generate_uuid()
        rns_path = self.resolve_rns_path(name)
        if rns_path.startswith("~"):
            return rns_path[2:]
        # For the purposes of building the path to the underlying data resource we don't need the slash
        return rns_path.lstrip("/")

    #########################
    # Folder Operations
    #########################

    def resolve_rns_path(self, path: str):
        if path == ".":
            return self.current_folder
        if path.startswith("./"):
            return self.current_folder + "/" + path[2:]
        # if path == '~':
        #     return '/rh'
        # if path.startswith('~/'):
        #     return '/rh/' + path[2:]
        # TODO break out paths for remote rns?
        if path == "@":
            return self.default_folder
        if path.startswith("@/"):
            return self.default_folder + "/" + path[2:]
        # if path == '^':
        #     return self.RH_BUILTINS_FOLDER
        # if path.startswith('^'):
        #     return self.RH_BUILTINS_FOLDER + '/' + path[1:]
        if path[0] not in ["/", "~", "^"]:
            return self.current_folder + "/" + path
        return path

    @staticmethod
    def split_rns_name_and_path(path: str):
        return Path(path).name, str(Path(path).parent)

    def exists(
        self,
        name_or_path,
        resource_type: str = None,
    ):
        config = self.load_config(name_or_path)
        if not config:
            return False
        if resource_type:
            return config.get("resource_type") == resource_type
        return True

    def locate(
        self,
        name,
        resolve_path=True,
    ):
        """Return the path for a resource."""
        # First check if name is in current folder

        if name == "/":
            return None

        if resolve_path:
            name = self.resolve_rns_path(name)

        if name.startswith("~"):
            return name.replace("~", self.rh_directory)

        if name.startswith("^"):
            return name.replace("^", self.rh_builtins_directory + "/")

        # TODO [DG] see if this breaks anything, also make it traverse the various rns folders to find the resource
        # if name.startswith('/'):
        #     if self.exists(name):
        #         return self.resource_uri(name)

        return None

    def set_folder(self, path: str, create=False):
        from runhouse.rns.folders.folder import Folder, folder

        if isinstance(path, Folder):
            abs_path = path.rns_address
        else:
            abs_path = self.resolve_rns_path(path)
            if abs_path in ["~", "~/"]:
                create = False
            if create:
                folder(path=path, dryrun=False)

        self._prev_folders += [self.current_folder]
        self.current_folder = abs_path

    def unset_folder(self):
        """Sort of like `cd -`, but with a full stack of the previous folder's set. Resets the
        current_folder to the previous one on the stack, the current_folder right before the
        current one was set."""
        if len(self._prev_folders) == 0:
            # TODO should we be raising an error here?
            return
        self.current_folder = self._prev_folders.pop(-1)

    def contents(self, name_or_path, full_paths):
        from runhouse.rns.folders.folder import folder

        folder_url = self.locate(name_or_path)
        return folder(name=name_or_path, path=folder_url).resources(
            full_paths=full_paths
        )
