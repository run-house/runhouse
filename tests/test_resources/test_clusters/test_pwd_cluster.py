import logging
import unittest

from tests.conftest import local_docker_cluster_passwd, password_cluster
from tests.test_obj_store import (
    test_cancel_run,  # noqa: F401
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

UNIT = {"cluster": [local_docker_cluster_passwd]}
LOCAL = {"cluster": [local_docker_cluster_passwd]}
MINIMAL = {"cluster": [password_cluster, local_docker_cluster_passwd]}


if __name__ == "__main__":
    unittest.main()
