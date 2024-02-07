import logging

from tests.conftest import docker_cluster_pwd_ssh_no_auth, password_cluster
from tests.test_obj_store import (
    test_get_from_cluster,  # noqa: F401
    test_multiprocessing_streaming,  # noqa: F401
    test_pinning_and_arg_replacement,  # noqa: F401
    test_pinning_in_memory,  # noqa: F401
    test_put_and_get_on_cluster,  # noqa: F401
    test_put_resource,  # noqa: F401
    test_stateful_generator,  # noqa: F401
    test_stream_logs,  # noqa: F401
    test_tqdm_streaming,  # noqa: F401
)

logger = logging.getLogger(__name__)

UNIT = {"cluster": [docker_cluster_pwd_ssh_no_auth]}
LOCAL = {"cluster": [docker_cluster_pwd_ssh_no_auth]}
MINIMAL = {"cluster": [password_cluster, docker_cluster_pwd_ssh_no_auth]}
