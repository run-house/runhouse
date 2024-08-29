import atexit

# Following design pattern for singleton variables from here:
# https://docs.python.org/3/faq/programming.html#how-do-i-share-global-variables-across-modules

from runhouse.rns.defaults import Defaults
from runhouse.rns.rns_client import RNSClient
from runhouse.servers.obj_store import ObjStore

# Configure the logger once
# TODO commenting out for now because this duplicates the logging config in the root logger
# logging.config.dictConfig(LOGGING_CONFIG)

configs = Defaults()

ssh_tunnel_cache = {}


def clean_up_ssh_connections():
    for _, v in ssh_tunnel_cache.items():
        v.terminate()


atexit.register(clean_up_ssh_connections)

rns_client = RNSClient(configs=configs)

# Note: this initalizes a dummy global object. The obj_store must
# be properly initialized by a servlet via initialize.
obj_store = ObjStore()
