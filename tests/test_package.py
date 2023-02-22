import unittest
from pathlib import Path

import runhouse as rh


def setup():
    pass


def test_from_string():
    p = rh.Package.from_string("reqs:~/runhouse")
    assert p.local_path == str(Path.home() / "runhouse")


@unittest.skip("Not yet implemented.")
def test_share():
    import shutil

    # Create a local temp folder to install for the package
    tmp_path = Path.cwd().parent / "tmp_package"
    tmp_path.mkdir(parents=True, exist_ok=True)
    for i in range(3):
        output_file = Path(f"{tmp_path}/sample_file_{i}.txt")
        output_file.write_text(f"file{i}")

    p = rh.Package.from_string("local:./tmp_package")
    p.name = "package_to_share"

    c = rh.cluster(name="/aabrahami/rh-cpu")
    p.to_cluster(dest_cluster=c)

    p.share(users=["josh@run.house", "donny@run.house"], access_type="write")

    shutil.rmtree(tmp_path)

    assert True


@unittest.skip("Not yet implemented.")
def test_reload():
    rh.exists("@/runhouse")
    package = rh.package(name="/jlewitt1/runhouse")
    print(package.config_for_rns)
    assert "s3://" in package.fsspec_url


if __name__ == "__main__":
    setup()
    unittest.main()
