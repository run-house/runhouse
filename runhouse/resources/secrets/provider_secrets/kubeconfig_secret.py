import copy
import os
from typing import Dict

import yaml

from runhouse.resources.secrets.provider_secrets.provider_secret import ProviderSecret
from runhouse.resources.secrets.utils import _check_file_for_mismatches
from runhouse.utils import create_local_dir


class KubeConfigSecret(ProviderSecret):
    """
    .. note::
        To create a KubeConfigSecret, please use the factory method :func:`provider_secret` with ``provider=="kubernetes"``.
    """

    _PROVIDER = "kubernetes"
    _DEFAULT_CREDENTIALS_PATH = "~/.kube/config"

    @staticmethod
    def from_config(config: dict, dryrun: bool = False, _resolve_children: bool = True):
        # try block if for the case we are trying to load a shared secret.
        return KubeConfigSecret(**config, dryrun=dryrun)

    def _from_path(self, path: str = None):
        path = path or self.path
        if not path:
            return {}

        path = os.path.expanduser(path)
        if os.path.exists(path):
            try:
                with open(path) as f:
                    contents = yaml.safe_load(f)
            except:
                contents = {}
            return contents
        return {}

    def _write_to_file(
        self,
        path: str,
        values: Dict,
        overwrite: bool = False,
        write_config: bool = True,
    ):
        new_secret = copy.deepcopy(self)
        path = path or self.path
        if not _check_file_for_mismatches(
            path, self._from_path(path), values, overwrite
        ):
            full_path = create_local_dir(path)
            with open(full_path, "w") as f:
                yaml.safe_dump(values, f)

            if write_config:
                self._add_to_rh_config(path)

        new_secret._values = None
        new_secret.path = path
        return new_secret
