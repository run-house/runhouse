import pprint
import sys
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from runhouse.globals import obj_store, rns_client
from runhouse.logger import get_logger

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

logger = get_logger(__name__)


class Resource:
    RESOURCE_TYPE = "resource"

    def __init__(
        self,
        name: Optional[str] = None,
        dryrun: bool = False,
        access_level: ResourceAccess = ResourceAccess.WRITE,
        visibility: ResourceVisibility = ResourceVisibility.PRIVATE,
        **kwargs,
    ):
        """
        Runhouse abstraction for objects that can be saved, shared, and reused.

        Args:
            name (Optional[str], optional): Name to assign the resource. (Default: None)
            dryrun (bool, optional): Whether to create the resource object, or load the object as a dryrun.
                (Default: ``False``)
            access_level (:obj:`ResourceAccess`, optional): Access level to provide for the resource.
                (Default: ``ResourceAccess.WRITE``)
            visibility (:obj:`ResourceVisibility`, optional): Type of visibility to provide for the resource.
                (Default: ``ResourceVisibility.PRIVATE``)
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

        self.dryrun = dryrun
        self.access_level = access_level
        self._visibility = visibility

    # TODO add a utility to allow a parameter to be specified as "default" and then use the default value

    @property
    def config_for_rns(self):
        # Added for BC for version 0.0.20
        return self.config(condensed=False)

    def config(self, condensed=True):
        config = {
            "name": self.rns_address or self.name,
            "resource_type": self.RESOURCE_TYPE,
            "resource_subtype": self.__class__.__name__,
        }
        self.save_attrs_to_config(
            config,
            [
                "visibility",  # Handles Enum to string conversion
            ],
        )
        return config

    def _resource_string_for_subconfig(
        self, resource: Union[None, str, "Resource"], condensed=True
    ):
        """Returns a string representation of a sub-resource for use in a config."""
        if resource is None or isinstance(resource, str):
            return resource
        if isinstance(resource, Resource):
            if condensed and resource.rns_address:
                # We operate on the assumption that rns_address is only populated once a resource has been saved.
                # That way, if rns_address is not None, we have reasonable likelihood that the resource was saved and
                # we can just pass the address. The only exception here is if the resource is a built-in.
                if resource.rns_address.startswith("^"):
                    # Fork the resource if it's a built-in and consider it a new resource
                    resource._rns_folder = None
                    return resource.config(condensed)
                return resource.rns_address
            else:
                # If the resource doesn't have an rns_address, we consider it unsaved and put the whole config into
                # the parent config.
                return resource.config(condensed)
        raise ValueError(
            f"Resource {resource} is not a valid sub-resource for {self.__class__.__name__}"
        )

    @property
    def rns_address(self):
        """Traverse up the filesystem until reaching one of the directories in rns_base_folders,
        then compute the relative path to that."""
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

    @property
    def visibility(self):
        return self._visibility

    @visibility.setter
    def visibility(self, visibility):
        self._visibility = visibility

    @rns_address.setter
    def rns_address(self, new_address):
        self.name = new_address  # Note, this saves the resource to the new address!

    def _save_sub_resources(self, folder: str = None):
        """Overload by child resources to save any resources they hold internally."""
        pass

    def pin(self):
        """Write the resource to the object store."""
        from runhouse.resources.hardware.utils import _current_cluster

        if _current_cluster():
            if obj_store.has_local_storage:
                obj_store.put_local(self._name, self)
            else:
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

    def save(self, name: str = None, overwrite: bool = True, folder: str = None):
        """Register the resource, saving it to local working_dir config and RNS config store. Uses the resource's
        `self.config()` to generate the dict to save."""

        # add this resource this run's downstream artifact registry if it's being saved as part of a run
        rns_client.add_downstream_resource(name or self.name)

        self._save_sub_resources(folder)
        if name:
            self.name = name

        # TODO handle self.access == 'read' instead of this weird overwrite argument
        save(self, overwrite=overwrite, folder=folder)

        return self

    def __str__(self):
        return pprint.pformat(self.config())

    @classmethod
    def _check_for_child_configs(cls, config: dict):
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
            elif isinstance(val, list):
                val = [str(item) if isinstance(item, int) else item for item in val]
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
    def from_name(
        cls,
        name: str,
        load_from_den: bool = True,
        dryrun: bool = False,
        _alt_options: Dict = None,
        _resolve_children: bool = True,
    ):
        """Load existing Resource via its name.

        Args:
            name (str): Name of the resource to load from name.
            load_from_den (bool, optional): Whether to try loading the module from Den. (Default: ``True``)
            dryrun (bool, optional): Whether to construct the object or load as dryrun. (Default: ``False``)
        """
        # TODO is this the right priority order?
        from runhouse.resources.hardware.utils import _current_cluster

        if _current_cluster() and obj_store.contains(name):
            return obj_store.get(name)

        config = rns_client.load_config(name=name, load_from_den=load_from_den)

        if _alt_options:
            config = cls._compare_config_with_alt_options(config, _alt_options)
            if not config:
                return None
        if not config:
            raise ValueError(f"Resource {name} not found.")
        config["name"] = name

        if _resolve_children:
            config = cls._check_for_child_configs(config)

        # Add this resource's name to the resource artifact registry if part of a run
        rns_client.add_upstream_resource(name)

        # Uses child class's from_config
        return cls.from_config(
            config=config, dryrun=dryrun, _resolve_children=_resolve_children
        )

    @staticmethod
    def from_config(config: Dict, dryrun: bool = False, _resolve_children: bool = True):
        """Load or construct resource from config.

        Args:
            config (Dict): Resource config.
            dryrun (bool, optional): Whether to construct resource or load as dryrun (Default: ``False``)
        """
        resource_type = config.pop("resource_type", None)
        dryrun = config.pop("dryrun", False) or dryrun

        if resource_type == "resource":
            return Resource(**config, dryrun=dryrun)

        resource_class = getattr(
            sys.modules["runhouse"], resource_type.capitalize(), None
        )
        if not resource_class:
            raise TypeError(f"Could not find module associated with {resource_type}")

        if _resolve_children:
            config = resource_class._check_for_child_configs(config)

        loaded = resource_class.from_config(
            config=config,
            dryrun=dryrun,
            _resolve_children=_resolve_children,
        )
        if loaded.name:
            rns_client.add_upstream_resource(loaded.name)
        return loaded

    def unname(self):
        """Remove the name of the resource. This changes the resource name to anonymous and deletes any local
        or RNS configs for the resource."""
        self.delete_configs()
        self._name = None

    def history(self, limit: int = None) -> List[Dict]:
        """Return the history of the resource, including specific config fields (e.g. folder path) and which runs
        have overwritten it.

        Args:
            limit (int, optional): If specified, return the last ``limit`` number of entries in the history.
                Otherwise, return the entire history. (Default: ``None``)
        """
        if not self.rns_address:
            raise ValueError("Resource must have a name in order to have a history")

        if self.rns_address[:2] == "~/":
            raise ValueError(
                "Resource must be saved to Den (not local) in order to have a history"
            )

        resource_uri = rns_client.resource_uri(self.rns_address)
        base_uri = f"{rns_client.api_server_url}/resource/history/{resource_uri}"
        uri = f"{base_uri}?limit={limit}" if limit else base_uri

        resp = rns_client.session.get(uri, headers=rns_client.request_headers())
        if resp.status_code != 200:
            logger.warning(
                f"Received [{resp.status_code}] from Den GET '{uri}': No resource history found: {load_resp_content(resp)}"
            )
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
                if isinstance(val, Enum):
                    val = val.value
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
        users: Union[str, List[str]] = None,
        access_level: Union[ResourceAccess, str] = ResourceAccess.READ,
        visibility: Optional[Union[ResourceVisibility, str]] = None,
        notify_users: bool = True,
        headers: Optional[Dict] = None,
    ) -> Tuple[Dict[str, ResourceAccess], Dict[str, ResourceAccess]]:
        """Grant access to the resource for a list of users (or a single user). By default, the user will
        receive an email notification of access (if they have a Runhouse account) or instructions on creating
        an account to access the resource. If ``visibility`` is set to ``public``, users will not be notified.

        .. note::
            You can only grant access to other users if you have write access to the resource.

        Args:
            users (Union[str, list], optional): Single user or list of user emails and / or runhouse account usernames.
                If none are provided and ``visibility`` is set to ``public``, resource will be made publicly
                available to all users. (Default: ``None``)
            access_level (:obj:`ResourceAccess`, optional): Access level to provide for the resource.
                (Default: ``read``).
            visibility (:obj:`ResourceVisibility`, optional): Type of visibility to provide for the shared
                resource. By default, the visibility is private. (Default: ``None``)
            notify_users (bool, optional): Whether to send an email notification to users who have been given access.
                Note: This is relevant for resources which are not ``shareable``. (Default: ``True``)
            headers (Dict, optional): Request headers to provide for the request to RNS. Contains the user's auth token.
                Example: ``{"Authorization": f"Bearer {token}"}``

        Returns:
            Tuple(Dict, Dict, Set):

            `added_users`:
                Users who already have a Runhouse account and have been granted access to the resource.
            `new_users`:
                Users who do not have Runhouse accounts and received notifications via their emails.
            `valid_users`:
                Set of valid usernames and emails from ``users`` parameter.

        Example:
            >>> # Write access to the resource for these specific users.
            >>> # Visibility will be set to private (users can search for and view resource in Den dashboard)
            >>> my_resource.share(users=["username1", "user2@gmail.com"], access_level='write')

            >>> # Make resource public, with read access to the resource for all users
            >>> my_resource.share(visibility='public')
        """
        if self.name is None:
            raise ValueError("Resource must have a name in order to share")

        if users is None and visibility is None:
            raise ValueError(
                "Must specify `visibility` for the resource if no users are provided."
            )

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

        if visibility is not None:
            # Update the resource in Den with this global visibility value
            self.visibility = visibility

            logger.debug(f"Updating resource with visibility: {self.visibility}")

        self.save()

        if isinstance(users, str):
            users = [users]

        added_users, new_users, valid_users = rns_client.grant_resource_access(
            rns_address=self.rns_address,
            user_emails=users,
            access_level=access_level,
            notify_users=notify_users,
            headers=headers,
        )
        return added_users, new_users, valid_users

    def revoke(
        self, users: Union[str, List[str]] = None, headers: Optional[Dict] = None
    ):
        """Revoke access to the resource.

        Args:
            users (Union[str, str], optional): List of user emails and / or runhouse account usernames
                (or a single user). If no users are specified will revoke access for all users. (Default: ``None``)
            headers (Optional[Dict]): Request headers to provide for the request to RNS. Contains the user's auth token.
                Example: ``{"Authorization": f"Bearer {token}"}``
        """
        if isinstance(users, str):
            users = [users]

        request_uri = rns_client.resource_uri(self.rns_address)
        uri = f"{rns_client.api_server_url}/resource/{request_uri}/users/access"
        resp = rns_client.session.put(
            uri,
            json={"users": users, "access_level": ResourceAccess.DENIED},
            headers=headers or rns_client.request_headers(),
        )

        if resp.status_code != 200:
            raise Exception(
                f"Received [{resp.status_code}] from Den PUT '{uri}': Failed to revoke access for resource: {load_resp_content(resp)}"
            )
