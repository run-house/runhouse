import contextlib

import pytest

from runhouse.globals import rns_client


@contextlib.contextmanager
def test_account():
    """Used for the purposes of testing resource sharing among different accounts.
    When inside the context manager, use the test account credentials before reverting back to the original
    account when exiting."""

    try:
        account = rns_client.load_account_from_env()
        if account is None:
            pytest.skip("`TEST_TOKEN` or `TEST_USERNAME` not set, skipping test.")
        yield account

    finally:
        rns_client.load_account_from_file()
