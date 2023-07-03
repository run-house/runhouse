# Following design pattern for singleton variables from here:
# https://docs.python.org/3/faq/programming.html#how-do-i-share-global-variables-across-modules
import logging.config

# from runhouse.logger import LOGGING_CONFIG
from runhouse.rns.defaults import Defaults
from runhouse.rns.obj_store import ObjStore
from runhouse.rns.rns_client import RNSClient

# Configure the logger once
# TODO commenting out for now because this duplicates the logging config in the root logger
# logging.config.dictConfig(LOGGING_CONFIG)

logger = logging.getLogger(__name__)

configs = Defaults()

open_cluster_tunnels = {}

rns_client = RNSClient(configs=configs)

obj_store = None
env_obj_store = None
try:
    import ray  # noqa: F401

    # Rename "cluster_obj_store"
    obj_store = ObjStore(name="cluster_obj_store")
    env_for_key = ObjStore(name="env_for_key")
except Exception:
    pass

env_servlets = {}
