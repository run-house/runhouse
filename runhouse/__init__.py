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

# Note these are global variables that are instantiated within rh_config.py:
from .rh_config import configs, obj_store, rns_client
from .rns.blobs import blob, Blob, file, File
from .rns.defaults import Defaults
from .rns.envs import conda_env, CondaEnv, env, Env
from .rns.folders import Folder, folder, GCSFolder, S3Folder
from .rns.function import function, Function
from .rns.hardware import (
    cluster,
    Cluster,
    ondemand_cluster,
    OnDemandCluster,
    sagemaker_cluster,
    SageMakerCluster,
)
from .rns.kvstores.kvstore import KVStore
from .rns.login import login, logout
from .rns.module import module, Module
from .rns.packages import git_package, GitPackage, package, Package
from .rns.queues.queue import queue, Queue
from .rns.run import Run, run, RunStatus, RunType
from .rns.secrets.secrets import Secrets
from .rns.tables.table import table, Table
from .rns.utils.runs import capture_stdout

# Briefly keep for BC.
send = function

# Syntactic sugar
fn = function

# This allows us to natively interact with resources in the object store from a python interpreter on the cluster
from runhouse.rns.utils.hardware import _current_cluster

if _current_cluster():
    import ray

    ray.init(ignore_reinit_error=True)
    obj_store.set_name("base")

__version__ = "0.0.12"
