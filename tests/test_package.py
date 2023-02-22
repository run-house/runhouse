import unittest
from pathlib import Path

import runhouse as rh


def setup():
    pass


def test_from_string():
    p = rh.Package.from_string("reqs:~/runhouse")
    assert p.local_path == str(Path.home() / "runhouse")


def test_share_package():
    import shutil

    # Create a local temp folder to install for the package
    tmp_path = Path.cwd().parent / "tmp_package"
    tmp_path.mkdir(parents=True, exist_ok=True)
    for i in range(3):
        output_file = Path(f"{tmp_path}/sample_file_{i}.txt")
        output_file.write_text(f"file{i}")

    p = rh.Package.from_string("local:./tmp_package")
    p.name = "package_to_share"  # shareable resource requires a name

    c = rh.cluster(name="/jlewitt1/rh-cpu")
    p.to_cluster(dest_cluster=c)

    p.share(users=["josh@run.house", "donny@run.house"], access_type="write")

    shutil.rmtree(tmp_path)

    # Confirm the package's folder is now on the cluster
    status_codes = c.run(commands=["ls tmp_package"])
    assert "sample_file_0.txt" in status_codes[0][1]


@unittest.skip("Not yet implemented.")
def test_share_git_package():
    pass
    # TODO [JL]
    # repo_package = rh.git_package(
    #     git_url=f"https://github.com/{username}/{repo_name}.git",
    #     revision=branch_name,
    # )


if __name__ == "__main__":
    setup()
    unittest.main()
