from .rns.send import send, Send
from .rns.secrets import Secrets
from .rns.defaults import Defaults
from .rns.login import login
# Note these are global variables that are instantiated within rh_config.py:
from .rh_config import configs, rns_client

from runhouse.rns.top_level_rns_fns import exists, set_folder, unset_folder, current_folder, \
    resources, ipython, locate, load, \
    pin_to_memory, get_pinned_object

from runhouse.rns.packages.package import package, Package
from .rns.folders.folder import folder, Folder
from .rns.tables.table import table, Table
from .rns.blob import blob, Blob
from .rns.hardware.skycluster import cluster, Cluster
from .rns.kvstores.kvstore import KVStore

# TODO [DG] do this properly
__version__ = '0.0.1.1'
