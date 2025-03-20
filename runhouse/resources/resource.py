import pprint
import sys
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Union

from runhouse.globals import obj_store, rns_client
from runhouse.logger import get_logger

from runhouse.rns.top_level_rns_fns import (
    resolve_rns_path,
    save,
    split_rns_name_and_path,
)
from runhouse.rns.utils.api import load_resp_content, read_resp_data, ResourceAccess
from runhouse.rns.utils.names import is_valid_resource_name

logger = get_logger(__name__)


class Resource:
    RESOURCE_TYPE = "resource"

    def __init__(
        self,
        name: Optional[str] = None,
        dryrun: bool = False,
        access_level: ResourceAccess = ResourceAccess.WRITE,
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
        """
        self._name, self._rns_folder = None, None
        if name is not None:
            if name.startswith("/builtins/"):
                name = name[len("/builtins/") :]
            if name[0] == "^" and name != "^":
                name = name[1:]
            # Validate that name complies with a simple regex
            if not is_valid_resource_name(name):
                raise ValueError(
                    f"Invalid name: {name} "
                    "Resource names are limited to alphanumerics, dashes, and underscores. Max 200 characters. "
                    "Slashes may be used to specify folders and must be included as the first character."
                )
            self._name, self._rns_folder = rns_client.split_rns_name_and_path(
                rns_client.resolve_rns_path(name)
            )

        self.dryrun = dryrun
        self.access_level = access_level

    # TODO add a utility to allow a parameter to be specified as "default" and then use the default value

    def config(self, condensed=True):
        config = {
            "name": self.rns_address or self.name,
            "resource_type": self.RESOURCE_TYPE,
            "resource_subtype": self.__class__.__name__,
        }
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
        """Full address of resource saved in Den. Has the format {username}/{resource_name}"""
        if (
            self.name is None or self._rns_folder is None
        ):  # Anonymous folders have no rns address
            return None

        return str(Path(self._rns_folder) / self.name)

    @property
    def name(self):
        """Resource name."""
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

    def _save_sub_resources(self, folder: str = None):
        """Overload by child resources to save any resources they hold internally."""
        pass

    def pin(self):
        """Write the resource to the object store."""
        from runhouse.resources.hardware.utils import _current_compute

        if _current_compute():
            if obj_store.has_local_storage:
                obj_store.put_local(self._name, self)
            else:
                obj_store.put(self._name, self)
        else:
            raise ValueError("Cannot pin a resource outside of a cluster.")

    def refresh(self):
        """Update the resource in the object store."""
        from runhouse.resources.hardware.utils import _current_compute

        if _current_compute():
            return obj_store.get(self._name)
        else:
            return self

    def save(self, name: str = None, overwrite: bool = True, folder: str = None):
        """Register the resource, saving it to the Den config store. Uses the resource's
        ``self.config()`` to generate the dict to save."""

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
    def from_name(
        cls,
        name: str,
        load_from_den: bool = True,
        dryrun: bool = False,
        _resolve_children: bool = True,
    ):
        """Load existing Resource via its name.

        Args:
            name (str): Name of the resource to load from name.
            load_from_den (bool, optional): Whether to try loading the module from Den. (Default: ``True``)
            dryrun (bool, optional): Whether to construct the object or load as dryrun. (Default: ``False``)
        """
        # TODO is this the right priority order?
        from runhouse.resources.hardware.utils import _current_compute

        if _current_compute() and obj_store.contains(name):
            return obj_store.get(name)

        config = rns_client.load_config(name=name, load_from_den=load_from_den)
        if not config:
            raise ValueError(f"Resource {name} not found.")

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
        """Remove the name of the resource. This changes the resource name to anonymous and deletes any Den configs
        for the resource."""
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
        """Delete the resource's config from Den config store."""
        rns_client.delete_configs(resource=self)

    def save_attrs_to_config(self, config: Dict, attrs: List[str]):
        """Save the given attributes to the config"""
        for attr in attrs:
            val = self.__getattribute__(attr)
            if val or (val is False):
                # allow for saving `False` but not other falsey types
                if isinstance(val, Enum):
                    val = val.value
                config[attr] = val

    def is_local(self):
        return (
            hasattr(self, "install_target")
            and isinstance(self.install_target, str)
            and self.install_target.startswith("~")
            or hasattr(self, "compute")
            and self.compute == "file"
        )
