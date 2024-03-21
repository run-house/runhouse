import os
import platform
import sys

try:
    from pip._internal.operations import freeze
except ImportError:  # pip < 10.0
    from pip.operations import freeze

py_version = sys.version.replace("\n", " ")
py_platform = platform.platform()

pkgs = freeze.freeze()
pip_pkgs = "\n".join(
    pkg
    for pkg in pkgs
    if any(
        name in pkg
        for name in {
            # runhouse
            "runhouse",
            # required installs
            "wheel",
            "rich",
            "fsspec",
            "pyarrow",
            "sshfs",
            "typer",
            "skypilot",
            "fastapi",
            "uvicorn",
            "pyOpenSSL"
            # aws
            "awscli",
            "boto3",
            "pycryptodome",
            "s3fs",
            "sshtunnel",
            # azure
            "azure-cli",
            "azure-core",
            # gcp
            "google-api-python-client",
            "google-cloud-storage",
            "gcsfs",
            # docker
            "docker",
        }
    )
)

print(f"Python Platform: {py_platform}")
print(f"Python Version: {py_version}")
print()
print(f"Relevant packages: \n{pip_pkgs}")
print()
os.system("sky check")
os.system("sky status --refresh")
