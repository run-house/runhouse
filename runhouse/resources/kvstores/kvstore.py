from typing import Any, Optional, Union

from runhouse import Cluster, Env
from runhouse.resources.module import Module


class Kvstore(Module):
    RESOURCE_TYPE = "kvstore"
    DEFAULT_CACHE_FOLDER = ".cache/runhouse/kvstores"

    """Simple dict wrapper to act as key-value/object storage. Wrapping this in an actor allows us to
    access it across Ray processes and nodes, and even keep some things pinned to Python memory."""

    def __init__(
        self,
        name: Optional[str] = None,
        system: Union[Cluster, str] = None,
        env: Optional[Env] = None,
        dryrun: bool = False,
        **kwargs,
    ):
        """
        Runhouse KVStore object

        .. note::
                To build a KVStore, please use the factory method :func:`kvstore`.
        """
        super().__init__(name=name, system=system, env=env, dryrun=dryrun, **kwargs)
        self.data = {}

    def put(self, key: str, value: Any):
        self.data[key] = value

    def get(self, key: str, default=None):
        if default == KeyError:
            return self.data[key]
        return self.data.get(key, default)

    def pop(self, key: str, *args):
        # We accept *args here to match the signature of dict.pop (throw an error if key is not found,
        # unless another arg is provided as a default)
        return self.data.pop(key, *args)

    def keys(self):
        return list(self.data.keys())

    def values(self):
        return list(self.data.values())

    def items(self):
        return list(self.data.items())

    def clear(self):
        self.data = {}

    def rename_key(self, old_key, new_key, *args):
        # We accept *args here to match the signature of dict.pop (throw an error if key is not found,
        # unless another arg is provided as a default)
        self.data[new_key] = self.data.pop(old_key, *args)

    def __len__(self):
        return len(self.data)

    def contains(self, key: str):
        return key in self

    def __contains__(self, key: str):
        return key in self.data

    def __getitem__(self, key: str):
        return self.data[key]

    def __setitem__(self, key: str, value: Any):
        self.data[key] = value

    def __delitem__(self, key: str):
        del self.data[key]

    def __repr__(self):
        return repr(self.data)
