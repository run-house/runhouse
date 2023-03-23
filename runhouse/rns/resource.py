import logging

import pprint
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import requests

from runhouse.rh_config import rns_client
from runhouse.rns.api_utils.resource_access import ResourceAccess
from runhouse.rns.api_utils.utils import load_resp_content, read_resp_data
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
            - Cluster :py:class:`.hardware.cluster.Cluster`
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
            "name": self.rns_address,
            "resource_type": self.RESOURCE_TYPE,
            "resource_subtype": self.__class__.__name__,
        }
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

    def _save_sub_resources(self):
        """Overload by child resources to save any resources they hold internally."""
        pass

    def save(
        self,
        name: str = None,
        overwrite: bool = True,
    ):
        """Register the resource, saving it to local working_dir config and RNS config store. Uses the resource's
        `self.config_for_rns` to generate the dict to save."""

        self._save_sub_resources()
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
        save(self, overwrite=overwrite)

        return self

    def __str__(self):
        return pprint.pformat(self.config_for_rns)

    @classmethod
    def _check_for_child_configs(cls, config):
        """Overload by child resources to load any resources they hold internally."""
        return config

    @classmethod
    def from_name(cls, name, dryrun=True):
        """Load existing Resource via its name."""
        config = rns_client.load_config(name=name)
        if not config:
            raise ValueError(f"Resource {name} not found.")
        config["name"] = name
        config = cls._check_for_child_configs(config)
        # Uses child class's from_config
        return cls.from_config(config=config, dryrun=dryrun)

    def unname(self):
        """Remove the name of the resource. This changes the resource name to anonymous and deletes any local
        or RNS configs for the resource."""
        self.delete_configs()
        self._name = None

    @staticmethod
    def history(name: str) -> List[Dict]:
        """Return the history of the resource, including specific config fields (e.g. blob path) and which runs
        have overwritten it."""
        resource_uri = rns_client.resource_uri(name)
        resp = requests.get(
            f"{rns_client.api_server_url}/resource/history/{resource_uri}",
            headers=rns_client.request_headers,
        )
        if resp.status_code != 200:
            raise Exception(
                f"Failed to load resource history: {load_resp_content(resp)}"
            )

        resource_history = read_resp_data(resp)
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
            and isinstance(self.install_target, str)
            and self.install_target.startswith("~")
            or hasattr(self, "system")
            and self.system == "file"
        )

    # TODO [DG] Implement proper sharing of subresources (with an overload of some kind)
    def share(
        self,
        users: list,
        access_type: Union[ResourceAccess, str] = ResourceAccess.READ,
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
            notify_users (bool): Send email notification to users who have been given access. Defaults to `False`.

        Returns:
            `added_users`: users who already have an account and have been granted access to the resource.
            `new_users`: users who do not have Runhouse accounts.

        Example:
            >>> added_users, new_users = my_resource.share(users=["username1", "user2@gmail.com"], access_type='write')
        """
        if self.name is None:
            raise ValueError("Resource must have a name in order to share")

        if hasattr(self, "system") and self.system in ["ssh", "sftp"]:
            logger.warning(
                "Sharing a resource located on a cluster is not recommended. For persistence, we suggest"
                "saving to a cloud storage system (ex: `s3` or `gs`). You can copy your cluster based "
                f"{self.RESOURCE_TYPE} to your desired storage provider using the `.to()` method. "
                f"For example: `{self.RESOURCE_TYPE}.to(system='rh-cpu')`"
            )

        if self.is_local():
            if self.RESOURCE_TYPE == "package":
                raise TypeError(
                    f"Unable to share a local {self.RESOURCE_TYPE}. Please make sure the {self.RESOURCE_TYPE} is "
                    f"located on a cluster. You can use the `.to_cluster()` method to do so. "
                    f"For example: `{self.name}.to_cluster(system='rh-cpu')`"
                )
            else:
                raise TypeError(
                    f"Unable to share a local {self.RESOURCE_TYPE}. Please make sure the {self.RESOURCE_TYPE} is "
                    f"located on a cluster or a remote system. You can use the `.to()` method to do so. "
                    f"For example: `{self.name}.to(system='s3')`"
                )

        if isinstance(access_type, str):
            access_type = ResourceAccess(access_type)

        if not rns_client.exists(self.rns_address):
            self.save(name=rns_client.local_to_remote_address(self.rns_address))

        added_users, new_users = rns_client.grant_resource_access(
            rns_address=self.rns_address,
            user_emails=users,
            access_type=access_type,
            notify_users=notify_users,
        )
        return added_users, new_users
