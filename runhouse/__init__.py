from runhouse.resources.blobs import blob, Blob, file, File
from runhouse.resources.envs import conda_env, CondaEnv, env, Env
from runhouse.resources.folders import Folder, folder, GCSFolder, S3Folder
from runhouse.resources.function import function, Function
from runhouse.resources.hardware import (
    _current_cluster,
    cluster,
    Cluster,
    ondemand_cluster,
    OnDemandCluster,
    sagemaker_cluster,
    SageMakerCluster,
    kubernetes_cluster,
    KubernetesCluster
)

# WARNING: Any built-in module that is imported here must be capitalized followed by all lowercase, or we will
# will not find the module class when attempting to reconstruct it from a config.
from runhouse.resources.kvstores.kvstore import Kvstore
from runhouse.resources.module import Module, module
from runhouse.resources.packages import git_package, GitPackage, package, Package
from runhouse.resources.provenance import capture_stdout, Run, run, RunStatus, RunType
from runhouse.resources.queues import Queue
from runhouse.resources.tables import Table, table

from runhouse.rns.secrets.secrets import Secrets

from runhouse.rns.top_level_rns_fns import (
    current_folder,
    exists,
    here,
    ipython,
    load,
    locate,
    resources,
    set_folder,
    unset_folder,
)

# Note these are global variables that are instantiated within globals.py:
from .globals import configs, obj_store

from .rns.login import login, logout

# Briefly keep for BC.
send = function

# Syntactic sugar
fn = function

# This allows us to natively interact with resources in the object store from a python interpreter on the cluster

if _current_cluster():
    import ray

    ray.init(ignore_reinit_error=True)
    obj_store.set_name("base")

__version__ = "0.0.13"
