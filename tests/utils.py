import contextlib
import importlib
import os
import uuid
from pathlib import Path

import pytest

import runhouse as rh
import yaml
from runhouse.constants import TESTING_LOG_LEVEL

from runhouse.globals import rns_client
from runhouse.servers.obj_store import get_cluster_servlet, ObjStore, RaySetupOption


def get_ray_worker_servlet_and_obj_store(env_name):
    """Helper method for getting object store"""

    test_obj_store = ObjStore()
    test_obj_store.initialize(env_name, setup_ray=RaySetupOption.GET_OR_FAIL)

    test_worker_servlet = test_obj_store.get_worker_servlet(
        env_name=env_name,
        create=True,
    )

    return test_worker_servlet, test_obj_store


def get_ray_cluster_servlet(cluster_config=None):
    """Helper method for getting base cluster servlet"""
    cluster_servlet = get_cluster_servlet(create_if_not_exists=True)

    if cluster_config:
        ObjStore.call_actor_method(
            cluster_servlet, "aset_cluster_config", cluster_config
        )

    return cluster_servlet


def get_pid_and_ray_node(a=0):
    import ray

    return (
        os.getpid(),
        ray.runtime_context.RuntimeContext(ray.worker.global_worker).get_node_id(),
    )


def get_random_str(length: int = 8):
    if length > 32:
        raise ValueError("Max length of random string is 32")

    return str(uuid.uuid4())[:length]


@contextlib.contextmanager
def friend_account():
    """Used for the purposes of testing resource sharing among different accounts.
    When inside the context manager, use the test account credentials before reverting back to the original
    account when exiting."""

    local_rh_package_path = Path(
        importlib.util.find_spec("runhouse").origin
    ).parent.parent
    dotenv_path = local_rh_package_path / ".env"
    if not dotenv_path.exists():
        dotenv_path = None  # Default to standard .env file search

    try:
        account = rns_client.load_account_from_env(
            token_env_var="KITCHEN_TESTER_TOKEN",
            usr_env_var="KITCHEN_TESTER_USERNAME",
            dotenv_path=dotenv_path,
        )
        if account is None:
            pytest.skip(
                "`KITCHEN_TESTER_TOKEN` or `KITCHEN_TESTER_USERNAME` not set, skipping test."
            )
        yield account

    finally:
        rns_client.load_account_from_file()


@contextlib.contextmanager
def friend_account_in_org():
    """Used for the purposes of testing resource sharing among different accounts.
    When inside the context manager, use the test account credentials before reverting back to the original
    account when exiting. This account is also included as part of the test org used in various tests"""

    local_rh_package_path = Path(
        importlib.util.find_spec("runhouse").origin
    ).parent.parent
    dotenv_path = local_rh_package_path / ".env"
    if not dotenv_path.exists():
        dotenv_path = None  # Default to standard .env file search

    try:
        account = rns_client.load_account_from_env(
            token_env_var="ORG_MEMBER_TOKEN",
            usr_env_var="ORG_MEMBER_USERNAME",
            dotenv_path=dotenv_path,
        )
        if account is None:
            pytest.skip(
                "`ORG_MEMBER_TOKEN` or `ORG_MEMBER_USERNAME` not set, skipping test."
            )
        yield account

    finally:
        rns_client.load_account_from_file()


def test_env(logged_in=False):
    return rh.env(
        reqs=["pytest", "httpx", "pytest_asyncio", "pandas", "numpy<=1.26.4"],
        working_dir=None,
        env_vars={"RH_LOG_LEVEL": os.getenv("RH_LOG_LEVEL") or TESTING_LOG_LEVEL},
        setup_cmds=[
            f"mkdir -p ~/.rh; touch ~/.rh/config.yaml; "
            f"echo '{yaml.safe_dump(rh.configs.defaults_cache)}' > ~/.rh/config.yaml"
        ]
        if logged_in
        else False,
    )


def remove_config_keys(config, keys_to_skip):
    for key in keys_to_skip:
        config.pop(key, None)
    return config
