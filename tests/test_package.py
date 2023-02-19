import unittest
from pathlib import Path

import runhouse as rh


def setup():
    pass


def test_from_string():
    p = rh.Package.from_string("reqs:~/runhouse")
    assert p.local_path == str(Path.home() / "runhouse")


def test_share():
    p = rh.Package.from_string("reqs:~/runhouse/runhouse")
    p.name = "package_to_share"
    added_users, new_users = p.share(
        users=["josh@run.house", "donny@run.house"], snapshot=False, access_type="write"
    )
    assert added_users or new_users

    p.delete_configs()


def test_reload():
    rh.exists("@/runhouse")
    package = rh.package(name="/donnyg/runhouse")
    print(package.config_for_rns)
    assert "s3://" in package.fsspec_url


if __name__ == "__main__":
    setup()
    unittest.main()
