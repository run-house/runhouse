import contextlib
import importlib
import uuid
from pathlib import Path

import pytest

import runhouse as rh
import yaml

from runhouse.globals import rns_client
from runhouse.servers.obj_store import ObjStore, RaySetupOption


def get_ray_servlet_and_obj_store(env_name):
    """Helper method for getting auth servlet and base env servlet"""

    test_obj_store = ObjStore()
    test_obj_store.initialize(env_name, setup_ray=RaySetupOption.TEST_PROCESS)

    servlet = ObjStore.get_env_servlet(
        env_name=env_name,
        create=True,
    )

    return servlet, test_obj_store


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
            token_env_var="TEST_TOKEN",
            usr_env_var="TEST_USERNAME",
            dotenv_path=dotenv_path,
        )
        if account is None:
            pytest.skip("`TEST_TOKEN` or `TEST_USERNAME` not set, skipping test.")
        yield account

    finally:
        rns_client.load_account_from_file()


def test_env(logged_in=False):
    return rh.env(
        reqs=["pytest", "httpx", "pytest_asyncio"],
        working_dir=None,
        setup_cmds=[
            f"mkdir -p ~/.rh; touch ~/.rh/config.yaml; "
            f"echo '{yaml.safe_dump(rh.configs.defaults_cache)}' > ~/.rh/config.yaml"
        ]
        if logged_in
        else False,
        name="base_env",
    )
