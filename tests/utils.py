import contextlib
import importlib
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import pytest
import requests

import runhouse as rh
import yaml

from runhouse.constants import DEFAULT_PROCESS_NAME
from runhouse.globals import rns_client

from runhouse.resources.hardware.utils import ClusterStatus, RunhouseDaemonStatus
from runhouse.servers.http.http_utils import CreateProcessParams
from runhouse.servers.obj_store import ObjStore, RaySetupOption

from tests.constants import DEFAULT_KEYPAIR_KEYPATH, TEST_ENV_VARS, TEST_REQS


def get_ray_servlet_and_obj_store(env_name):
    """Helper method for getting object store"""

    test_obj_store = ObjStore()
    test_obj_store.initialize(env_name, setup_ray=RaySetupOption.GET_OR_FAIL)

    test_servlet = test_obj_store.get_servlet(
        name=env_name,
        create_process_params=CreateProcessParams(name=env_name),
        create=True,
    )

    return test_servlet, test_obj_store


def get_pid_and_ray_node(a=0):
    import logging

    import ray

    pid = os.getpid()
    node_id = ray.runtime_context.RuntimeContext(ray.worker.global_worker).get_node_id()

    print(f"PID: {pid}")
    logging.info(f"Node ID: {node_id}")

    return pid, node_id


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
        account = rns_client.load_account_from_env(
            token_env_var="RH_TOKEN",
            usr_env_var="RH_USERNAME",
            dotenv_path=dotenv_path,
        )
        if account is None:
            pytest.skip("`RH_USERNAME` or `RH_TOKEN` not set, skipping test.")
        yield account

    finally:
        os.environ["RH_USERNAME"] = original_username
        rns_client.load_account_from_file()


def setup_test_base(cluster, logged_in=False):
    setup_cmds = [
        f"mkdir -p ~/.rh; touch ~/.rh/config.yaml; "
        f"echo '{yaml.safe_dump(rh.configs.defaults_cache)}' > ~/.rh/config.yaml"
    ]

    cluster.pip_install(TEST_REQS)
    cluster.set_process_env_vars(DEFAULT_PROCESS_NAME, TEST_ENV_VARS)
    if logged_in:
        cluster.run_bash(setup_cmds)


def keep_config_keys(config, keys_to_keep):
    condensed_config = {key: config.get(key) for key in keys_to_keep}
    return condensed_config


def get_default_keypair_path():
    if rh.configs.get("default_ssh_key"):
        secret = rh.secret(rh.configs.get("default_ssh_key"))
        key_path = secret.path
    else:
        key_path = DEFAULT_KEYPAIR_KEYPATH
    return str(Path(key_path).expanduser())


def set_daemon_and_cluster_status(
    cluster: rh.Cluster,
    daemon_status: RunhouseDaemonStatus,
    cluster_status: ClusterStatus,
):
    cluster_uri = rh.globals.rns_client.format_rns_address(cluster.rns_address)
    headers = rh.globals.rns_client.request_headers()
    api_server_url = rh.globals.rns_client.api_server_url

    # Note: the resource includes the cluster's status and the runhouse daemon status
    status_data_resource = {
        "daemon_status": daemon_status,
        "cluster_status": cluster_status,
        "cluster_status_last_checked": datetime.utcnow().isoformat(),
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


def _get_env_var_value(env_var):
    import os

    return os.environ[env_var]


####################################################################################################
# ray utils
####################################################################################################
def init_remote_cluster_servlet_actor(
    current_ip: str,
    runtime_env: Optional[Dict] = None,
    cluster_config: Optional[Dict] = None,
    servlet_name: Optional[str] = "cluster_servlet",
):
    import ray
    from runhouse.servers.cluster_servlet import ClusterServlet

    remote_actor = (
        ray.remote(ClusterServlet)
        .options(
            name=servlet_name,
            get_if_exists=True,
            lifetime="detached",
            namespace="runhouse",
            max_concurrency=1000,
            resources={f"node:{current_ip}": 0.001},
            num_cpus=0,
            runtime_env=runtime_env,
        )
        .remote(cluster_config=cluster_config, name=servlet_name)
    )
    return remote_actor


####################################################################################################
# general utils
####################################################################################################
def compare_python_versions(cluster_version, local_version):
    """used for comparing cluster py version vs. local py version."""
    # Split the version strings into a list of integers
    cluster_version_parts = list(map(int, cluster_version.split(".")))
    local_version_parts = list(map(int, local_version.split(".")))

    # Compare the major, minor, and patch versions
    for cv, lv in zip(cluster_version_parts, local_version_parts):
        if cv > lv:
            return cluster_version
        elif cv < lv:
            return local_version

    # If all parts are equal, return nothing
    return None
