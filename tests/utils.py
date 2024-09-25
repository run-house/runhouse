import contextlib
import importlib
import json
import os
import uuid
from datetime import datetime
from pathlib import Path

import pytest
import requests

import runhouse as rh
import yaml
from runhouse.constants import TESTING_LOG_LEVEL

from runhouse.globals import rns_client

from runhouse.resources.hardware.utils import ResourceServerStatus
from runhouse.servers.obj_store import get_cluster_servlet, ObjStore, RaySetupOption


def get_ray_env_servlet_and_obj_store(env_name):
    """Helper method for getting object store"""

    test_obj_store = ObjStore()
    test_obj_store.initialize(env_name, setup_ray=RaySetupOption.GET_OR_FAIL)

    test_env_servlet = test_obj_store.get_env_servlet(
        env_name=env_name,
        create=True,
    )

    return test_env_servlet, test_obj_store


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


@contextlib.contextmanager
def org_friend_account(new_username: str, token: str, original_username: str):
    """Used for the purposes of testing listing clusters associated with test-org"""

    os.environ["RH_USERNAME"] = new_username
    os.environ["RH_TOKEN"] = token

    local_rh_package_path = Path(
        importlib.util.find_spec("runhouse").origin
    ).parent.parent
    dotenv_path = local_rh_package_path / ".env"
    if not dotenv_path.exists():
        dotenv_path = None  # Default to standard .env file search

    try:
        account = rns_client.load_account_from_env(dotenv_path=dotenv_path)
        if account is None:
            pytest.skip("`RH_USERNAME` or `RH_TOKEN` not set, skipping test.")
        yield account

    finally:
        os.environ["RH_USERNAME"] = original_username
        rns_client.load_account_from_file()


def test_env(logged_in=False):
    return rh.env(
        reqs=["pytest", "httpx", "pytest_asyncio", "pandas", "numpy<=1.26.4"],
        working_dir=None,
        env_vars={
            "RH_LOG_LEVEL": os.getenv("RH_LOG_LEVEL") or TESTING_LOG_LEVEL,
            "RH_AUTOSTOP_INTERVAL": os.getenv("RH_AUTOSTOP_INTERVAL"),
        },
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


def set_cluster_status(cluster: rh.Cluster, status: ResourceServerStatus):
    cluster_uri = rh.globals.rns_client.format_rns_address(cluster.rns_address)
    headers = rh.globals.rns_client.request_headers()
    api_server_url = rh.globals.rns_client.api_server_url

    # updating the resource collection as well, because the cluster.list() gets the info from the resource
    status_data_resource = {
        "status": status,
        "status_last_checked": datetime.utcnow().isoformat(),
    }
    requests.put(
        f"{api_server_url}/resource/{cluster_uri}",
        data=json.dumps(status_data_resource),
        headers=headers,
    )


def set_output_env_vars():
    env = os.environ.copy()
    # Set the COLUMNS and LINES environment variables to control terminal width and height,
    # so we could get the runhouse cluster list output properly using subprocess
    env["COLUMNS"] = "250"
    env["LINES"] = "40"  # Set a height value, though COLUMNS is the key one here

    return env
