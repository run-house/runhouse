import os
import unittest

import pytest

import runhouse as rh

from runhouse.globals import configs


@pytest.mark.rnstest
def test_download_defaults():
    rh.globals.configs.defaults_cache["default_folder"] = "nonsense"
    local_defaults = rh.configs.load_defaults_from_file()
    local_defaults.pop("secrets")
    rh.configs.upload_defaults(defaults=local_defaults)
    loaded_defaults = rh.configs.download_defaults()
    assert local_defaults == loaded_defaults
    assert rh.globals.rns_client.default_folder == "nonsense"


@pytest.mark.rnstest
def test_opentelemetry_otlp_params_are_defined():
    otlp_endpoint_url = os.getenv("OTLP_ENDPOINT_URL") or configs.get(
        "otlp_endpoint_url"
    )
    otlp_username = os.getenv("OTLP_USERNAME") or configs.get("otlp_username")
    otlp_password = os.getenv("OTLP_PASSWORD") or configs.get("otlp_password")

    assert (
        otlp_endpoint_url
    ), "No otlp_endpoint_url provided. Either set `OTLP_ENDPOINT_URL` env variable "
    "or set `otlp_endpoint_url` in the .rh config file"

    assert (
        otlp_username
    ), "No otlp_username provided. Either set `OTLP_USERNAME` env variable "
    "or set `otlp_username` in the .rh config file"

    assert (
        otlp_password
    ), "No otlp_password provided. Either set `OTLP_PASSWORD` env variable "
    "or set `otlp_password` in the .rh config file"


if __name__ == "__main__":
    unittest.main()
