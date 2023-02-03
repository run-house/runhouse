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
    # Can share via email or username
    p.share(users=["jlewitt1", "donnyg"], snapshot=True, access_type="write")


def test_reload():
    rh.exists("@/runhouse")
    package = rh.package(name="/donnyg/runhouse")
    print(package.config_for_rns)
    assert "s3://" in package.fsspec_url


if __name__ == "__main__":
    setup()
    unittest.main()
