import runhouse as rh


def test_download_defaults():
    rh.globals.configs.defaults_cache["default_folder"] = "nonsense"
    local_defaults = rh.configs.load_defaults_from_file()
    local_defaults.pop("secrets")
    rh.configs.upload_defaults(defaults=local_defaults)
    loaded_defaults = rh.configs.download_defaults()
    assert local_defaults == loaded_defaults
    assert rh.globals.rns_client.default_folder == "nonsense"
