import importlib
import json
import os
import shutil
from pathlib import Path
from typing import Any, Dict, Optional, Set, Union

import dotenv

import requests
from pydantic import BaseModel

from runhouse.logger import get_logger

from runhouse.rns.utils.api import (
    generate_uuid,
    load_resp_content,
    read_resp_data,
    remove_null_values_from_dict,
    ResourceAccess,
)

from runhouse.utils import locate_working_dir

logger = get_logger(__name__)


# This is a copy of the Pydantic model that we use to validate in Den
class ResourceStatusData(BaseModel):
    cluster_config: dict
    server_cpu_utilization: float
    server_gpu_utilization: Optional[float]
    server_memory_usage: Dict[str, Any]
    server_gpu_usage: Optional[Dict[str, Any]]
    env_servlet_processes: Dict[str, Dict[str, Any]]
    server_pid: int
    runhouse_version: str


class RNSClient:
    """Manage a particular resource with the runhouse database"""

    CORE_RNS_FIELDS = [
        "name",
        "resource_type",
        "visibility",
        "folder",
        "users",
    ]
    DEFAULT_FS = "file"

    def __init__(self, configs) -> None:
        self.run_stack = []
        self._configs = configs
        self._prev_folders = []

        self.rh_directory = str(Path(locate_working_dir()) / "rh")
        self.rh_builtins_directory = str(
            Path(importlib.util.find_spec("runhouse").origin).parent / "builtins"
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
            str(Path(importlib.util.find_spec("runhouse").origin).parent / "builtins")
        )
        self._index_base_folders(rns_base_folders)
        self._current_folder = None

        self.session = requests.Session()

    @property
    def default_folder(self):
        return self._configs.default_folder

    @property
    def current_folder(self):
        return self._current_folder if self._current_folder else self.default_folder

    @current_folder.setter
    def current_folder(self, value):
        self._current_folder = value

    @property
    def token(self):
        return self._configs.token

    @property
    def username(self):
        return self._configs.get("username", None)

    @property
    def api_server_url(self):
        url_as_env_var = os.getenv("API_SERVER_URL")
        if url_as_env_var:
            return url_as_env_var
        return self._configs.get("api_server_url", None)

    @property
    def autosave(self):
        return self._configs.get("autosave", True)

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
    def base_folder(resource_uri: str):
        """Top level folder associated with a resource."""
        if resource_uri is None:
            return None

        paths = resource_uri.split("/")
        return paths[1] if len(paths) > 1 else None

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

    def autosave_resources(self):
        return bool(self.autosave and self.token)

    def request_headers(
        self, resource_address: str = None, headers: dict = None
    ) -> Union[dict, None]:
        """Returns the authentication headers to use for requests made to Den or to a cluster.

        If the request is being made to Den, we simply construct the request headers with the user's existing
        Runhouse token.

        If the request is being made to (or from) a cluster, we generate a new unique token to prevent exposing the
        user's original Runhouse token on the cluster. This new token is based on the user's existing Den token and
        the Den address of the resource (or cluster API) they are attempting to access.

        For example, if userA tries to access a function on a cluster that was shared with them by userB, we generate a
        new token containing userA's Den token and top level directory associated with the
        resource (e.g. if the function has a rns address of: "/userB/some_func", the top level directory will be set
        to: "userB", indicating that the function's owner/namespace is userB).

        The updated token used in requests made by userA to the cluster would then look like:
        "hash(userA den token + resource top level directory) + resource top level directory + userA", where
        "hash" is a sha256 hash function.

        This method also ensures that each user <> resource relationship will yield the same token, allowing
        resource owners to more easily identify specific users consuming or accessing their resource.

        Args:
            resource_address (str, optional): Name of the top level directory or namespace that the resource
                is associated with. For example, if the Den address of a function is: "/userA/some_func", the resource
                address would be "userA". If not provided, we construct headers with the user's default Runhouse token.
            headers (dict, optional): Request headers to use for the request. If not provided, we use the default
                headers (i.e. use the default Runhouse token).

        Returns:
            request_headers (Union[dict, None]): The resulting request headers, or ``None`` if no headers
                are required for the request.

        """
        if headers == {}:
            # Support use case where we explicitly do not want to provide headers (e.g. requesting a cert)
            return None

        if headers is None:
            # Use the default headers (i.e. the user's original Den token)
            headers: dict = self._configs.request_headers

        if not headers:
            return None

        if "Authorization" not in headers:
            raise ValueError(
                "Invalid request headers provided, expected in format: {Authorization: Bearer <token>}"
            )

        if resource_address is None:
            # If a base dir is not specified assume we are not sending a request to a cluster
            return headers

        den_token = None
        auth_header = headers["Authorization"]
        token_parts = auth_header.split(" ")
        if len(token_parts) == 2 and token_parts[0] == "Bearer":
            den_token = token_parts[1]

        if den_token is None:
            raise ValueError(
                "Failed to extract token from request auth header. Expected in format: Bearer <token>"
            )

        hashed_token = self.cluster_token(resource_address)

        return {"Authorization": f"Bearer {hashed_token}"}

    def cluster_token(
        self, resource_address: str, username: str = None, den_token: str = None
    ):
        """Load the hashed token as generated in Den. Cache the token value in-memory for a given resource address.
        Optionally provide a username and den token instead of using the default values stored in local configs."""
        if resource_address and "/" in resource_address:
            # If provided as a full rns address, extract the top level directory
            resource_address = self.base_folder(resource_address)

        uri = f"{self.api_server_url}/auth/token/cluster"
        token_payload = {
            "resource_address": resource_address,
            "username": username or self._configs.username,
        }

        headers = (
            {"Authorization": f"Bearer {den_token}"}
            if den_token
            else self._configs.request_headers
        )
        resp = self.session.post(
            uri,
            data=json.dumps(token_payload),
            headers=headers,
        )
        if resp.status_code != 200:
            raise Exception(
                f"Received [{resp.status_code}] from Den POST '{uri}': Failed to load cluster token: {load_resp_content(resp)}"
            )

        resp_data = read_resp_data(resp)
        return resp_data.get("token")

    def validate_cluster_token(self, cluster_token: str, cluster_uri: str) -> bool:
        """Checks whether a particular cluster token is valid for the given cluster URI"""
        request_uri = self.resource_uri(cluster_uri)
        uri = f"{self.api_server_url}/auth/token/cluster/{request_uri}"
        resp = self.session.get(
            uri,
            headers={"Authorization": f"Bearer {cluster_token}"},
        )
        return resp.status_code == 200

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

    def load_account_from_env(
        self, token_env_var="RH_TOKEN", usr_env_var="RH_USERNAME", dotenv_path=None
    ) -> Dict[str, str]:
        dotenv.load_dotenv(dotenv_path=dotenv_path)

        test_token = os.getenv(token_env_var)
        test_username = os.getenv(usr_env_var)
        if not (test_token and test_username):
            return None

        self._configs.token = test_token
        self._configs.username = test_username
        self._configs.default_folder = f"/{test_username}"

        # The client caches the folder that is used as the current folder variable, we clear this so it loads the new
        # folder when we switch accounts
        self._current_folder = None

        return {
            "token": self._configs.token,
            "username": self._configs.username,
            "default_folder": self._configs.default_folder,
        }

    def load_account_from_file(self) -> None:
        # Setting this to None causes it to be loaded from file upon next access
        self._configs.defaults_cache = None

        # Calling with .get explicitly loads from the config.yaml file
        self._configs.token = self._configs.get("token", None)
        self._configs.username = self._configs.get("username", None)
        self._configs.default_folder = self._configs.get("default_folder", None)

        # Same as above, for this to correctly load the account/folder from the new cache, it needs to be unset
        self._current_folder = None

    # Run Stack
    # ---------------------

    def start_run(self, run_obj: "Run"):
        self.run_stack.append(run_obj)

    def stop_run(self):
        return self.run_stack.pop()

    def current_run(self):
        if not self.run_stack:
            return None
        return self.run_stack[-1]

    def add_upstream_resource(self, name: str):
        """Add a resource's name to the current run's upstream artifact registry if it's being loaded"""
        current_run = self.current_run()
        if current_run:
            artifact_name = self.resolve_rns_path(name)
            current_run._register_upstream_artifact(artifact_name)

    def add_downstream_resource(self, name: str):
        """Add a resource's name to the current run's downstream artifact registry if it's being saved"""
        current_run = self.current_run()
        if current_run:
            artifact_name = self.resolve_rns_path(name)
            current_run._register_downstream_artifact(artifact_name)

    # ---------------------

    def grant_resource_access(
        self,
        *,
        rns_address: str,
        user_emails: list = None,
        access_level: ResourceAccess,
        notify_users: bool,
        headers: Optional[dict] = None,
    ):
        resource_uri = self.resource_uri(rns_address)
        headers = headers or self.request_headers()
        access_payload = {
            "users": user_emails,
            "access_level": access_level,
            "notify_users": notify_users,
        }
        uri = f"{self.api_server_url}/resource/{resource_uri}/users/access"
        resp = self.session.put(
            uri,
            data=json.dumps(access_payload),
            headers=headers,
        )
        if resp.status_code != 200:
            raise Exception(
                f"Received [{resp.status_code}] from Den PUT '{uri}': Failed to grant access to: {load_resp_content(resp)}"
            )

        resp_data: dict = read_resp_data(resp)
        added_users: dict = resp_data.get("added_users", {})
        new_users: dict = resp_data.get("new_users", {})
        valid_users: Set = resp_data.get("valid_users", set())

        return added_users, new_users, valid_users

    def load_config(
        self,
        name,
        load_from_den=True,
    ) -> dict:
        if not name:
            return {}

        from runhouse.resources.hardware.utils import _current_cluster

        if "/" not in name:
            name = f"{self.current_folder}/{name}"

        rns_address = self.resolve_rns_path(name)

        if rns_address == _current_cluster("name"):
            return _current_cluster("config")

        if rns_address[0] in ["~", "^"]:
            config = self._load_config_from_local(rns_address)
            if config:
                return config

        if load_from_den and rns_address.startswith("/"):
            request_headers = self.request_headers()
            if not request_headers:
                raise PermissionError(
                    "No Runhouse token provided. Try running `$ runhouse login` or visiting "
                    "https://run.house/login to retrieve a token. If calling via HTTP, please "
                    "provide a valid token in the Authorization header."
                )

            resource_uri = self.resource_uri(name)
            logger.debug(f"Attempting to load config for {rns_address} from RNS.")
            uri = f"{self.api_server_url}/resource/{resource_uri}"
            resp = self.session.get(
                uri,
                headers=request_headers,
            )
            if resp.status_code != 200:
                logger.debug(
                    f"Received [{resp.status_code}] from Den GET '{uri}': No config found in RNS: {load_resp_content(resp)}"
                )
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
        # TODO should we handle remote filesystems, or throw an error if system != 'file'?
        if not path:
            path = self.locate(rns_address, resolve_path=False)
            if not path:
                return None
        config_path = Path(path) / "config.json"
        if not config_path.exists():
            return None

        logger.debug(f"Loading config from local file {config_path}")
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
        `self.config()` to generate the dict to save."""
        rns_address = resource.rns_address
        config = resource.config()

        if not overwrite and self.exists(rns_address):
            raise ValueError(
                f"Resource {rns_address} already exists and overwrite is False."
            )

        if rns_address is None:
            raise ValueError("A resource must have a name to be saved.")

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
        # TODO [CC]: can maybe asterik out sensitive info instead of this approach
        if not config.get("ssh_creds"):
            logger.info(f"Saving config to RNS: {config}")
        else:
            logger.info(f"Saving config for {resource_name} to RNS")

        resource_uri = self.resource_uri(resource_name)
        put_uri = f"{self.api_server_url}/resource/{resource_uri}"

        payload = self.resource_request_payload(config)
        headers = self.request_headers()
        resp = self.session.put(put_uri, data=json.dumps(payload), headers=headers)
        if resp.status_code == 200:
            logger.debug(f"Config updated in Den for resource: {resource_uri}")
        elif resp.status_code == 422:  # No changes made to existing Resource
            logger.debug(
                f"Config for {resource_uri} has not changed, nothing to update"
            )
        elif resp.status_code == 404:  # Resource not found
            logger.debug(f"Saving new resource in Den for resource: {resource_uri}")
            # Resource does not yet exist, in which case we need to create from scratch
            post_uri = f"{self.api_server_url}/resource"
            resp = self.session.post(
                post_uri,
                data=json.dumps(payload),
                headers=headers,
            )
            if resp.status_code != 200:
                raise Exception(
                    f"Received [{resp.status_code}] from Den POST '{post_uri}': Failed to create new resource '{resource_uri}': {load_resp_content(resp)}"
                )
        else:
            raise Exception(
                f"Received [{resp.status_code}] from Den PUT '{put_uri}': Failed to save resource '{resource_uri}': {load_resp_content(resp)}"
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
            uri = f"{self.api_server_url}/resource/{resource_uri}"
            resp = self.session.delete(uri, headers=self.request_headers())
            if resp.status_code != 200:
                logger.error(
                    f"Received [{resp.status_code}] from Den DELETE '{uri}': Failed to delete resource: {load_resp_content(resp)}"
                )
            else:
                logger.info(f"Successful Den Delete '{uri}'")

    def resolve_rns_data_resource_name(self, name: str):
        """If no name is explicitly provided for the data resource, we need to create one based on the relevant
        rns path. If name is None, return a hex uuid.
        For example: my_func -> my_username/my_func"
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
            return self.current_folder + "/" + path[len("./") :]
        # if path == '~':
        #     return '/rh'
        # if path.startswith('~/'):
        #     return '/rh/' + path[2:]
        # TODO break out paths for remote rns?
        if path == "@":
            return self.default_folder
        if path.startswith("@/"):
            return self.default_folder + "/" + path[len("@/") :]
        # if path == '^':
        #     return self.RH_BUILTINS_FOLDER
        # if path.startswith('^'):
        #     return self.RH_BUILTINS_FOLDER + '/' + path[1:]
        return path

    @staticmethod
    def split_rns_name_and_path(path: str):
        if "/" not in path:
            return path, None
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
        from runhouse.resources.folders import Folder, folder

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
        from runhouse.resources.folders import folder

        folder_url = self.locate(name_or_path)
        return folder(name=name_or_path, path=folder_url).resources(
            full_paths=full_paths
        )
