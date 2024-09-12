from runhouse.resources.asgi import Asgi, asgi
from runhouse.resources.envs import conda_env, CondaEnv, env, Env
from runhouse.resources.folders import Folder, folder, GCSFolder, S3Folder
from runhouse.resources.functions.aws_lambda import LambdaFunction
from runhouse.resources.functions.aws_lambda_factory import aws_lambda_fn
from runhouse.resources.functions.function import Function
from runhouse.resources.functions.function_factory import function
from runhouse.resources.hardware import (
    cluster,
    Cluster,
    kubernetes_cluster,
    ondemand_cluster,
    OnDemandCluster,
)

# WARNING: Any built-in module that is imported here must be capitalized followed by all lowercase, or we will
# will not find the module class when attempting to reconstruct it from a config.
from runhouse.resources.module import Module, module
from runhouse.resources.packages import (
    CodeSyncError,
    git_package,
    GitPackage,
    package,
    Package,
)
from runhouse.resources.resource import Resource
from runhouse.resources.secrets import provider_secret, ProviderSecret, Secret, secret

from runhouse.rns.top_level_rns_fns import (
    as_caller,
    current_folder,
    exists,
    get_local_cluster_object,
    ipython,
    load,
    locate,
    set_folder,
    unset_folder,
)
from runhouse.utils import sync_function

# Note these are global variables that are instantiated within globals.py:
from .globals import configs, obj_store

from .rns.login import login, logout

# Briefly keep for BC.
send = function

# Syntactic sugar
fn = function


def __getattr__(name):
    if name == "here":
        # If it's either the first time or the cluster was not initialized before, attempt to retrieve the cluster again
        return sync_function(get_local_cluster_object)()

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__version__ = "0.0.35"
