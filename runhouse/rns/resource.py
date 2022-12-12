from typing import Optional, List
import json
import requests
import logging
from pathlib import Path
from typing import Tuple, Dict, Union

from ray import cloudpickle as pickle

from runhouse.rh_config import rns_client
from runhouse.rns.api_utils.resource_access import ResourceAccess
from runhouse.rns.api_utils.utils import read_response_data
from runhouse.rns.top_level_rns_fns import resolve_rns_path, split_rns_name_and_path, save

logger = logging.getLogger(__name__)


class Resource:
    RESOURCE_TYPE = None

    def __init__(self,
                 name: Optional[str] = None,
                 load_from: Optional[List[str]] = None,
                 save_to: Optional[List[str]] = None,
                 dryrun: Optional[bool] = None,
                 ):
        self._name, self._rns_folder = None, None
        if name is not None:
            # TODO validate that name complies with a simple regex
            self._name, self._rns_folder = rns_client.split_rns_name_and_path(
                rns_client.resolve_rns_path(name))

        self.save_to = save_to
        self.load_from = load_from
        self.dryrun = dryrun

    # TODO add a utility to allow a parameter to be specified as "default" and then use the default value

    @property
    def config_for_rns(self):
        config = {'type': self.RESOURCE_TYPE}
        config_attrs = ['name', 'rns_address']
        self.save_attrs_to_config(config, config_attrs)
        return config

    @staticmethod
    def is_picklable(obj) -> bool:
        try:
            pickle.dumps(obj)
        except pickle.PicklingError:
            return False
        return True

    def _resource_string_for_subconfig(self, resource):
        """Returns a string representation of a sub-resource for use in a config."""
        # TODO [DG] add __str__ method to resource class and package to prevent things
        #  like saving a package called "torch==1.12"
        if resource is None or isinstance(resource, str):
            return resource
        if resource.name:
            if resource.rns_address.startswith('/builtins/'):
                # Calls save internally and puts the resource in the current folder
                resource.name = resource.rns_address[10:]
            resource.save(save_to=self.save_to)
            return resource.rns_address
        return resource.config_for_rns

    @property
    def rns_address(self):
        """ Traverse up the filesystem until reaching one of the directories in rns_base_folders,
        then compute the relative path to that.

        Maybe later, account for folders along the path with a differnt RNS name."""

        if self.name is None or self._rns_folder is None:  # Anonymous folders have no rns address
            return None

        return str(Path(self._rns_folder) / self.name)

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, name):
        # Split the name and rns path if path is given (concat with current_folder if just stem is given)
        self._name, self._rns_folder = split_rns_name_and_path(resolve_rns_path(name))
        # self._rns_folder = rns_parent
        # self._name = name
        # self.save()

    @rns_address.setter
    def rns_address(self, new_address):
        self.name = new_address  # Note, this saves the resource to the new address!

    def save(self,
             name: str = None,
             save_to: Optional[List[str]] = None,
             snapshot: bool = False,
             overwrite: bool = True,
             **kwargs):
        """Register the resource, saving it to local working_dir config and RNS config store. Uses the resource's
        `self.config_for_rns` to generate the dict to save."""

        # TODO deal with logic of saving anonymous folder for the first time after naming, i.e.
        # Path(tempfile.gettempdir()).relative_to(self.url) ...
        if name:
            if '/' in name[1:]:
                self._name, self._rns_folder = split_rns_name_and_path(resolve_rns_path(name))
            else:
                self._name = name

        # TODO handle self.access == 'read' instead of this weird overwrite argument
        save(self,
             save_to=save_to if save_to is not None else self.save_to,
             snapshot=snapshot,
             overwrite=overwrite, **kwargs)

    def unname(self):
        """ Change the naming of the resource to anonymous and delete any local or RNS configs for the resource."""
        self.delete_configs()
        self._name = None

    def history(self, entries: int = 10) -> List[Dict]:
        """Return the history of this resource, including specific config fields (e.g. blob URL) and which runs
        have overwritten it."""
        resource_uri = rns_client.resource_uri(self.name)
        resp = requests.get(f'{rns_client.api_server_url}/resource/history/{resource_uri}?num_entries={entries}',
                            headers=rns_client.request_headers)
        if resp.status_code != 200:
            raise Exception(f'Failed to load resource history: {json.loads(resp.content)}')

        resource_history: list = read_response_data(resp)
        return resource_history

    # TODO delete sub-resources
    def delete_configs(self, delete_from: [Optional[str]] = None):
        """Delete the resource's config from local working_dir and RNS config store."""
        rns_client.delete_configs(resource=self, delete_from=delete_from)

    def save_attrs_to_config(self, config, attrs):
        for attr in attrs:
            val = self.__getattribute__(attr)
            if val:
                config[attr] = val

    # TODO [DG] Implement proper sharing of subresources (with an overload of some kind)
    def share(self,
              users: list,
              access_type: Union[ResourceAccess, str] = ResourceAccess.read
              ) -> Tuple[Dict[str, ResourceAccess], Dict[str, ResourceAccess]]:
        """Grant access to the resource for list of users. If a user has a Runhouse account they
        will receive an email notifying them of their new access. If the user does not have a Runhouse account they will
        also receive instructions on creating one, after which they will be able to have access to the Resource.
        Note: You can only grant resource access to other users if you have Write / Read privileges for the Resource.

        Args:
            users (list): list of user emails and / or runhouse account usernames.
            access_type (:obj:`ResourceAccess`, optional): access type to provide for the resource.

        Example:
            .. code-block:: python

               added_users, new_users = my_send.share(users=["username1", "user@gmail.com"], access_type='read')

        Returns:
            Tuple[Dict[str, ResourceAccess], Dict[str, ResourceAccess]]: Tuple of two dictionaries.

            `added_users`: users who already have an account and have been granted access to the resource.

            `new_users`: users who do not have Runhouse accounts.

        """
        if isinstance(access_type, str):
            access_type = ResourceAccess(access_type)

        if not rns_client.exists(self.rns_address, load_from=['rns']):
            self.save(save_to=['rns'])
        added_users, new_users = rns_client.grant_resource_access(resource_name=self.name,
                                                                  user_emails=users,
                                                                  access_type=access_type)
        return added_users, new_users
