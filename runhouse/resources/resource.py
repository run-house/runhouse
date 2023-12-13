import logging

import pprint
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import requests

from runhouse.globals import obj_store, rns_client
from runhouse.rns.top_level_rns_fns import (
    resolve_rns_path,
    save,
    split_rns_name_and_path,
)
from runhouse.rns.utils.api import (
    load_resp_content,
    read_resp_data,
    ResourceAccess,
    ResourceVisibility,
)

logger = logging.getLogger(__name__)


class Resource:
    RESOURCE_TYPE = "resource"

    def __init__(
        self,
        name: Optional[str] = None,
        dryrun: bool = False,
        provenance=None,
        access_level: Optional[ResourceAccess] = ResourceAccess.WRITE,
        global_visibility: Optional[ResourceVisibility] = ResourceVisibility.PRIVATE,
        **kwargs,
    ):
        """
        Runhouse abstraction for objects that can be saved, shared, and reused.

        Runhouse currently supports the following builtin Resource types:

        - Compute Abstractions
            - Cluster :py:class:`.hardware.cluster.Cluster`
            - Function :py:class:`.function.Function`
            - Module :py:class:`.module.Module`
            - Package :py:class:`.packages.package.Package`
            - Env: :py:class:`.envs.env.Env`


        - Data Abstractions
            - Blob :py:class:`.blob.Blob`
            - Folder :py:class:`.folders.folder.Folder`
            - Table :py:class:`.tables.table.Table`
        """
        self._name, self._rns_folder = None, None
        if name is not None:
            # TODO validate that name complies with a simple regex
            if name.startswith("/builtins/"):
                name = name[len("/builtins/") :]
            if name[0] == "^" and name != "^":
                name = name[1:]
            self._name, self._rns_folder = rns_client.split_rns_name_and_path(
                rns_client.resolve_rns_path(name)
            )

        from runhouse.resources.provenance import Run

        self.dryrun = dryrun
        # dryrun is true here so we don't spend time calling check on the server
        # if we're just loading down the resource (e.g. with .remote)
        self.provenance = (
            Run.from_config(provenance, dryrun=True)
            if isinstance(provenance, Dict)
            else provenance
        )
        self.access_level = access_level
        self.global_visibility = global_visibility

    # TODO add a utility to allow a parameter to be specified as "default" and then use the default value

    @property
    def config_for_rns(self):
        config = {
            "name": self.rns_address,
            "resource_type": self.RESOURCE_TYPE,
            "resource_subtype": self.__class__.__name__,
            "provenance": self.provenance.config_for_rns if self.provenance else None,
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

        Maybe later, account for folders along the path with a different RNS name."""

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
        if name is None:
            self._name = None
        else:
            self._name, self._rns_folder = split_rns_name_and_path(
                resolve_rns_path(name)
            )

    @rns_address.setter
    def rns_address(self, new_address):
        self.name = new_address  # Note, this saves the resource to the new address!

    def _save_sub_resources(self):
        """Overload by child resources to save any resources they hold internally."""
        pass

    def pin(self):
        """Write the resource to the object store."""
        from runhouse.resources.hardware.utils import _current_cluster

        if _current_cluster():
            obj_store.put(self._name, self)
        else:
            raise ValueError("Cannot pin a resource outside of a cluster.")

    def refresh(self):
        """Update the resource in the object store."""
        from runhouse.resources.hardware.utils import _current_cluster

        if _current_cluster():
            return obj_store.get(self._name)
        else:
            return self

    def save(
        self,
        name: str = None,
        overwrite: bool = True,
    ):
        """Register the resource, saving it to local working_dir config and RNS config store. Uses the resource's
        `self.config_for_rns` to generate the dict to save."""

        # add this resource this run's downstream artifact registry if it's being saved as part of a run
        rns_client.add_downstream_resource(name or self.name)

        self._save_sub_resources()
        if name:
            self.name = name

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
    def _compare_config_with_alt_options(cls, config, alt_options):
        """Overload by child resources to compare their config with the alt_options. If the user specifies alternate
        options, compare the config with the options. It's generally up to the child class to decide how to handle the
        options, but default behavior is provided. The default behavior simply checks if any of the alt_options are
        present in the config (with awareness of resources), and if their values differ, return None.

        If the child class returns None, it's deciding to override the config
        with the options. If the child class returns a config, it's deciding to use the config and ignore the options
        (or somehow incorporate them, rarely). Note that if alt_options are provided and the config is not found,
        no error is raised, while if alt_options are not provided and the config is not found, an error is raised.

        """

        def str_dict_or_resource_to_str(val):
            if isinstance(val, Resource):
                return val.rns_address
            elif isinstance(val, dict):
                # This can either be a sub-resource which hasn't been converted to a resource yet, or an
                # actual user-provided dict
                if "rns_address" in val:
                    return val["rns_address"]
                if "name" in val:
                    # convert a user-provided name to an rns_address
                    return rns_client.resolve_rns_path(val["name"])
                else:
                    return val
            else:
                return val

        for key, value in alt_options.items():
            if key in config:
                if str_dict_or_resource_to_str(value) != str_dict_or_resource_to_str(
                    config[key]
                ):
                    return None
            else:
                return None
        return config

    @classmethod
    def from_name(cls, name, dryrun=False, alt_options=None):
        """Load existing Resource via its name."""
        # TODO is this the right priority order?
        from runhouse.resources.hardware.utils import _current_cluster

        if _current_cluster() and obj_store.contains(name):
            return obj_store.get(name, check_other_envs=True)

        config = rns_client.load_config(name=name)

        if alt_options:
            config = cls._compare_config_with_alt_options(config, alt_options)
            if not config:
                return None
        if not config:
            raise ValueError(f"Resource {name} not found.")
        config["name"] = name
        config = cls._check_for_child_configs(config)

        # Add this resource's name to the resource artifact registry if part of a run
        rns_client.add_upstream_resource(name)

        # Uses child class's from_config
        return cls.from_config(config=config, dryrun=dryrun)

    @staticmethod
    def from_config(config, dryrun=False):
        resource_type = config.pop("resource_type")
        dryrun = config.pop("dryrun", False) or dryrun

        if resource_type == "resource":
            return Resource(**config, dryrun=dryrun)

        resource_class = getattr(
            sys.modules["runhouse"], resource_type.capitalize(), None
        )
        if not resource_class:
            raise TypeError(f"Could not find module associated with {resource_type}")
        config = resource_class._check_for_child_configs(config)

        loaded = resource_class.from_config(config=config, dryrun=dryrun)
        if loaded.name:
            rns_client.add_upstream_resource(loaded.name)
        return loaded

    def unname(self):
        """Remove the name of the resource. This changes the resource name to anonymous and deletes any local
        or RNS configs for the resource."""
        self.delete_configs()
        self._name = None

    def history(self, num_entries: int = None) -> List[Dict]:
        """Return the history of the resource, including specific config fields (e.g. blob path) and which runs
        have overwritten it."""
        if not self.rns_address:
            raise ValueError("Resource must have a name in order to have a history")

        if self.rns_address[:2] == "~/":
            raise ValueError(
                "Resource must be saved to Den (not local) in order to have a history"
            )

        resource_uri = rns_client.resource_uri(self.rns_address)
        base_uri = f"{rns_client.api_server_url}/resource/history/{resource_uri}"
        uri = f"{base_uri}?num_entries={num_entries}" if num_entries else base_uri

        resp = requests.get(uri, headers=rns_client.request_headers)
        if resp.status_code != 200:
            logger.warning(f"No resource history found: {load_resp_content(resp)}")
            return []

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
            if val is not None:
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
        users: Union[str, List[str]],
        access_level: Union[ResourceAccess, str] = ResourceAccess.READ,
        global_visibility: Optional[ResourceVisibility] = None,
        notify_users: bool = True,
        headers: Optional[Dict] = None,
    ) -> Tuple[Dict[str, ResourceAccess], Dict[str, ResourceAccess]]:
        """Grant access to the resource for the list of users (or a single user). If a user has a Runhouse account they
        will receive an email notifying them of their new access. If the user does not have a Runhouse account they will
        also receive instructions on creating one, after which they will be able to have access to the Resource.

        .. note::
            You can only grant resource access to other users if you have Write / Read privileges for the Resource.

        Args:
            users (list or str): list of user emails and / or runhouse account usernames (or a single user).
            access_level (:obj:`ResourceAccess`, optional): access level to provide for the resource.
            notify_users (bool): Send email notification to users who have been given access. Defaults to `False`.
            headers (Optional[Dict]): Request headers to provide for the request to RNS. Contains the user's auth token.
                Example: ``{"Authorization": f"Bearer {token}"}``

        Returns:
            Tuple(Dict, Dict):

            `added_users`:
                users who already have an account and have been granted access to the resource.
            `new_users`:
                users who do not have Runhouse accounts.

        Example:
            >>> added_users, new_users = my_resource.share(users=["username1", "user2@gmail.com"], access_level='write')
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
                    f"located on a cluster. You can use the `.to()` method to do so. "
                    f"For example: `{self.name}.to(system='rh-cpu')`"
                )
            else:
                raise TypeError(
                    f"Unable to share a local {self.RESOURCE_TYPE}. Please make sure the {self.RESOURCE_TYPE} is "
                    f"located on a cluster or a remote system. You can use the `.to()` method to do so. "
                    f"For example: `{self.name}.to(system='s3')`"
                )

        if isinstance(access_level, str):
            access_level = ResourceAccess(access_level)

        if global_visibility is not None:
            self.global_visibility = global_visibility
        self.save()

        if isinstance(users, str):
            users = [users]

        added_users, new_users = rns_client.grant_resource_access(
            rns_address=self.rns_address,
            user_emails=users,
            access_level=access_level,
            notify_users=notify_users,
            headers=headers,
        )
        return added_users, new_users
