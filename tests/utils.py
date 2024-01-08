import contextlib
import pkgutil
from pathlib import Path

import pytest

from runhouse.globals import rns_client
from runhouse.servers.http.http_server import HTTPServer
from runhouse.servers.obj_store import ObjStore


def get_ray_servlet(env_name):
    """Helper method for getting auth servlet and base env servlet"""
    import ray

    ray.init(
        ignore_reinit_error=True,
        runtime_env=None,
        namespace="runhouse",
    )

    servlet = HTTPServer.get_env_servlet(
        env_name=env_name,
        create=True,
        runtime_env=None,
    )

    return servlet


def get_test_obj_store(env_servlet_name: str):
    # Ensure servlet is running
    _ = get_ray_servlet(env_servlet_name)

    test_obj_store = ObjStore()
    test_obj_store.initialize(env_servlet_name)

    return test_obj_store


@contextlib.contextmanager
def test_account():
    """Used for the purposes of testing resource sharing among different accounts.
    When inside the context manager, use the test account credentials before reverting back to the original
    account when exiting."""

    local_rh_package_path = Path(pkgutil.get_loader("runhouse").path).parent.parent
    dotenv_path = local_rh_package_path / ".env"
    if not dotenv_path.exists():
        dotenv_path = None  # Default to standard .env file search

    try:
        account = rns_client.load_account_from_env(
            token_env_var="TEST_TOKEN",
            usr_env_var="TEST_USERNAME",
            dotenv_path=dotenv_path,
        )
        if account is None:
            pytest.skip("`TEST_TOKEN` or `TEST_USERNAME` not set, skipping test.")
        yield account

    finally:
        rns_client.load_account_from_file()
