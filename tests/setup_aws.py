import os

import runhouse as rh

token = os.getenv("TEST_TOKEN")
headers = {"Authorization": f"Bearer {token}"}

rh.Secrets.put(
    provider="sky",
    secret={
        "ssh_private_key": os.getenv("TEST_SKY_PRIVATE_KEY"),
        "ssh_public_key": os.getenv("TEST_SKY_PUBLIC_KEY"),
    },
    headers=headers,
)
rh.Secrets.put(
    provider="aws",
    secret={
        "access_key": os.getenv("TEST_AWS_ACCESS_KEY"),
        "secret_key": os.getenv("TEST_AWS_SECRET_KEY"),
    },
    headers=headers,
)

rh.login(token=token, download_secrets=True)
