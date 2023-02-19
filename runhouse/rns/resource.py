import json
import logging

import pprint
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import fsspec
import requests

from runhouse.rh_config import configs, rns_client
from runhouse.rns.api_utils.resource_access import ResourceAccess
from runhouse.rns.api_utils.utils import read_response_data
from runhouse.rns.top_level_rns_fns import (
    resolve_rns_path,
    save,
    split_rns_name_and_path,
)

logger = logging.getLogger(__name__)


class Resource:
    RESOURCE_TYPE = None

    def __init__(
        self,
        name: Optional[str] = None,
        dryrun: bool = None,
    ):
        """
        Runhouse abstraction for objects that can be saved, shared, and reused.

        Runhouse currently supports the following builtin Resource types:

        - Compute Abstractions
            - Cluster :py:class:`.system.cluster.Cluster`
            - Function :py:class:`.function.Function`
            - Package :py:class:`.packages.package.Package`


        - Data Abstractions
            - Blob :py:class:`.blob.Blob`
            - Folder :py:class:`.folders.folder.Folder`
            - Table :py:class:`.tables.table.Table`
        """
        self._name, self._rns_folder = None, None
        if name is not None:
            # TODO validate that name complies with a simple regex
            if name.startswith("/builtins/"):
                name = name[10:]
            if name[0] == "^" and name != "^":
                name = name[1:]
            self._name, self._rns_folder = rns_client.split_rns_name_and_path(
                rns_client.resolve_rns_path(name)
            )

        self.dryrun = dryrun

    # TODO add a utility to allow a parameter to be specified as "default" and then use the default value

    @property
    def config_for_rns(self):
        config = {
            "resource_type": self.RESOURCE_TYPE,
            "resource_subtype": self.__class__.__name__,
        }
        config_attrs = ["name", "rns_address"]
        self.save_attrs_to_config(config, config_attrs)
        return config

    def _resource_string_for_subconfig(self, resource):
        """Returns a string representation of a sub-resource for use in a config."""
        if resource is None or isinstance(resource, str):
            return resource
        if resource.name:
            if resource.rns_address.startswith("^"):
                # Calls save internally and puts the resource in the current folder
                resource.name = rns_client.resolve_rns_path(resource.rns_address[1:])
            return resource.rns_address
        return resource.config_for_rns

    @property
    def rns_address(self):
        """Traverse up the filesystem until reaching one of the directories in rns_base_folders,
        then compute the relative path to that.

        Maybe later, account for folders along the path with a differnt RNS name."""

        if (
            self.name is None or self._rns_folder is None
        ):  # Anonymous folders have no rns address
            return None

        return str(Path(self._rns_folder) / self.name)

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, name):
        # Split the name and rns path if path is given (concat with current_folder if just stem is given)
        self._name, self._rns_folder = split_rns_name_and_path(resolve_rns_path(name))

    @rns_address.setter
    def rns_address(self, new_address):
        self.name = new_address  # Note, this saves the resource to the new address!

    def save(
        self,
        name: str = None,
        snapshot: bool = False,
        overwrite: bool = True,
        **snapshot_kwargs,
    ):
        """Register the resource, saving it to local working_dir config and RNS config store. Uses the resource's
        `self.config_for_rns` to generate the dict to save."""

        # TODO deal with logic of saving anonymous folder for the first time after naming, i.e.
        # Path(tempfile.gettempdir()).relative_to(self.path) ...
        if name:
            if "/" in name[1:] or self._rns_folder is None:
                self._name, self._rns_folder = split_rns_name_and_path(
                    resolve_rns_path(name)
                )
            else:
                self._name = name

        # TODO handle self.access == 'read' instead of this weird overwrite argument
        save(self, snapshot=snapshot, overwrite=overwrite, **snapshot_kwargs)

        return self

    def __str__(self):
        return pprint.pformat(self.config_for_rns)

    @classmethod
    def from_name(cls, name, dryrun=False):
        config = rns_client.load_config(name=name)
        if not config:
            raise ValueError(f"Resource {name} not found.")
        config["name"] = name
        # Uses child class's from_config
        return cls.from_config(config=config, dryrun=dryrun)

    def unname(self):
        """Remove the name of the resource. This changes the resource name to anonymous and deletes any local
        or RNS configs for the resource."""
        self.delete_configs()
        self._name = None

    @staticmethod
    def history(name: str, entries: int = 10) -> List[Dict]:
        """Return the history of the resource, including specific config fields (e.g. blob path) and which runs
        have overwritten it."""
        resource_uri = rns_client.resource_uri(name)
        resp = requests.get(
            f"{rns_client.api_server_url}/resource/history/{resource_uri}",
            headers=rns_client.request_headers,
        )
        if resp.status_code != 200:
            raise Exception(
                f"Failed to load resource history: {json.loads(resp.content)}"
            )

        resource_history = read_response_data(resp)
        return resource_history

    # TODO delete sub-resources
    def delete_configs(self):
        """Delete the resource's config from local working_dir and RNS config store."""
        rns_client.delete_configs(resource=self)

    def save_attrs_to_config(self, config: Dict, attrs: List[str]):
        """Save the given attributes to the config"""
        for attr in attrs:
            val = self.__getattribute__(attr)
            if val:
                config[attr] = val

    def is_local(self):
        return (
            hasattr(self, "install_target")
            and self.install_target.startswith("~")
            or hasattr(self, "system")
            and self.system == "file"
        )

    def create_snapshot_resource(
        self,
        snapshot_system: str = None,
        snapshot_compression: str = None,
        snapshot_path: str = None,
    ):
        from runhouse.rns.folders.folder import PROVIDER_FS_LOOKUP

        system = snapshot_system or PROVIDER_FS_LOOKUP[configs.get("default_storage")]
        if system not in fsspec.available_protocols():
            raise ValueError(
                f"Invalid mount_fs: {snapshot_system}. Must be one of {fsspec.available_protocols()}"
            )
        if snapshot_compression not in fsspec.available_compressions():
            raise ValueError(
                f"Invalid mount_compression: {snapshot_compression}. Must be one of "
                f"{fsspec.available_compressions()}"
            )
        data_config = (
            {"compression": snapshot_compression} if snapshot_compression else {}
        )

        if not hasattr(self, "to"):
            raise AttributeError(
                f"Unable to snapshot and copy a local {self.RESOURCE_TYPE} to {system}. Resource "
                f"must have a `.to()` method in order to snapshot."
            )

        snapshot_resource = self.to(
            system=system, path=snapshot_path, data_config=data_config
        )

        rns_address = rns_client.local_to_remote_address(self.rns_address)
        snapshot_resource.save(name=rns_address)

        return snapshot_resource

    # TODO [DG] Implement proper sharing of subresources (with an overload of some kind)
    def share(
        self,
        users: list,
        access_type: Union[ResourceAccess, str] = ResourceAccess.read,
        snapshot: bool = True,
        snapshot_system: str = None,
        snapshot_compression: str = None,
        snapshot_path: str = None,
        notify_users: bool = False,
    ) -> Tuple[Dict[str, ResourceAccess], Dict[str, ResourceAccess]]:
        """Grant access to the resource for the list of users. If a user has a Runhouse account they
        will receive an email notifying them of their new access. If the user does not have a Runhouse account they will
        also receive instructions on creating one, after which they will be able to have access to the Resource.

        .. note::
            You can only grant resource access to other users if you have Write / Read privileges for the Resource.

        Args:
            users (list): list of user emails and / or runhouse account usernames.
            access_type (:obj:`ResourceAccess`, optional): access type to provide for the resource.
            snapshot (bool): Whether to create a snapshot of the resource. Defaults to `True`.
            snapshot_system (:obj: str, optional): Which system to use for the snapshot.
                See `fsspec.available_protocols()` for options. Defaults to `None`.
            snapshot_compression (:obj: str, optional): Compression to use for the snapshot.
                See `fsspec.available_compressions()` for options. Defaults to `None`.
            snapshot_path (:obj: str, optional): Specific path to use for the snapshot. Defaults to `None`.
            notify_users (bool): Send email notification to users who have been given access. Defaults to `False`.


        Returns:
            `added_users`: users who already have an account and have been granted access to the resource.
            `new_users`: users who do not have Runhouse accounts.

        Example:
            >>> added_users, new_users = my_resource.share(users=["username1", "user@gmail.com"], access_type='write')
        """
        if self.name is None:
            raise ValueError("Resource must have a name in order to share")

        if self.is_local() and snapshot:
            snapshot_resource = self.create_snapshot_resource()
            return snapshot_resource.share(
                users=users, access_type=access_type, snapshot=False
            )

        if isinstance(access_type, str):
            access_type = ResourceAccess(access_type)

        if not rns_client.exists(self.rns_address):
            self.save(name=rns_client.local_to_remote_address(self.rns_address))

        added_users, new_users = rns_client.grant_resource_access(
            resource_name=self.name,
            user_emails=users,
            access_type=access_type,
            notify_users=notify_users,
        )
        return added_users, new_users
