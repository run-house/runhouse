import os

import pytest

from kubetorch.globals import config

from .utils import summer


@pytest.mark.level("unit")
def test_config_load_username_from_env_var():
    """Test helper function that loads username from the environment variable"""
    os.environ["KT_USERNAME"] = "test-user"
    assert config._get_env_var("username") == "test-user"


@pytest.mark.level("unit")
def test_config_set_and_get_username():
    """Test that the username is loaded from the config file"""
    org_username = config.username

    try:
        # Set username
        config.set("username", "test-user")
        assert config.username == "test-user"

        config.set("username", "test-user-2")
        assert config.username == "test-user-2"
    except Exception as e:
        raise e
    finally:
        # Restore original config
        config.set("username", org_username)

    assert config.username == org_username


@pytest.mark.level("unit")
def test_config_set_invalid_username():
    """Test that the username is invalid if it is too long or contains invalid characters"""
    config.set("username", "test-user-2" * 10)
    assert config.username == "test-user-2test"

    config.set("username", "01-test-user")
    assert config.username == "test-user"

    config.set("username", "test.user-2")
    assert config.username == "test-user-2"

    with pytest.raises(ValueError):
        config.set("username", "usern@me")


@pytest.mark.level("unit")
def test_config_username_set_on_module():
    """Test that the username is set on the module"""
    import kubetorch as kt

    org_username = config.username

    try:
        config.set("username", "test-user")
        assert config.username == "test-user"

        fn = kt.fn(summer, name="summer")
        assert fn.service_name == "test-user-summer"
    except Exception as e:
        raise e
    finally:
        config.set("username", org_username)

    assert config.username == org_username


@pytest.mark.level("unit")
def test_config_prefix_username_default_true():
    """prefix_username defaults to True when unset."""
    org = config._prefix_username
    try:
        os.environ.pop("KT_PREFIX_USERNAME", None)
        config._prefix_username = None
        assert config.prefix_username is True
    finally:
        config._prefix_username = org


@pytest.mark.level("unit")
def test_config_prefix_username_from_env_var():
    """prefix_username reads the KT_PREFIX_USERNAME env var."""
    org = config._prefix_username
    try:
        os.environ["KT_PREFIX_USERNAME"] = "false"
        config._prefix_username = None
        assert config.prefix_username is False
    finally:
        os.environ.pop("KT_PREFIX_USERNAME", None)
        config._prefix_username = org


@pytest.mark.level("unit")
def test_config_set_prefix_username():
    """prefix_username can be set and validated via config.set."""
    org = config._prefix_username
    try:
        config.set("prefix_username", False)
        assert config.prefix_username is False
        config.set("prefix_username", "true")
        assert config.prefix_username is True
        with pytest.raises(ValueError):
            config.set("prefix_username", "maybe")
    finally:
        config._prefix_username = org


@pytest.mark.level("unit")
def test_config_set_prefix_username_none_unsets():
    """Setting prefix_username to None unsets it; the property re-resolves to the default."""
    org = config._prefix_username
    try:
        os.environ.pop("KT_PREFIX_USERNAME", None)
        config.set("prefix_username", False)
        assert config.prefix_username is False
        config.set("prefix_username", None)
        assert config._prefix_username is None
        assert config.prefix_username is True
    finally:
        config._prefix_username = org


@pytest.mark.level("unit")
def test_service_name_prefix_username_attr_false():
    """Setting prefix_username=False on a module yields a bare service name."""
    import kubetorch as kt

    org = config.username
    try:
        config.set("username", "test-user")
        f = kt.fn(summer, name="summer")
        f.prefix_username = False
        assert f.service_name == "summer"
    finally:
        config.set("username", org)


@pytest.mark.level("unit")
def test_service_name_prefix_username_config_false():
    """Global config prefix_username=False yields a bare service name."""
    import kubetorch as kt

    org_user = config.username
    org_prefix = config._prefix_username
    try:
        config.set("username", "test-user")
        config.set("prefix_username", False)
        f = kt.fn(summer, name="summer")
        assert f.service_name == "summer"
    finally:
        config.set("username", org_user)
        config._prefix_username = org_prefix


@pytest.mark.level("unit")
def test_service_name_prefix_username_invalidates_cache():
    """Changing prefix_username after service_name was accessed re-resolves the name."""
    import kubetorch as kt

    org = config.username
    try:
        config.set("username", "test-user")
        f = kt.fn(summer, name="summer")
        assert f.service_name == "test-user-summer"  # populates the cache
        f.prefix_username = False
        assert f.service_name == "summer"  # cache invalidated, re-resolved to bare name
    finally:
        config.set("username", org)


@pytest.mark.level("unit")
def test_fn_prefix_username_param_false():
    """fn(prefix_username=False) yields a bare service name."""
    import kubetorch as kt

    org = config.username
    try:
        config.set("username", "test-user")
        f = kt.fn(summer, name="summer", prefix_username=False)
        assert f.service_name == "summer"
    finally:
        config.set("username", org)


@pytest.mark.level("unit")
def test_fn_prefix_username_param_overrides_config():
    """Per-call prefix_username=True overrides global config False."""
    import kubetorch as kt

    org_user = config.username
    org_prefix = config._prefix_username
    try:
        config.set("username", "test-user")
        config.set("prefix_username", False)
        f = kt.fn(summer, name="summer", prefix_username=True)
        assert f.service_name == "test-user-summer"
    finally:
        config.set("username", org_user)
        config._prefix_username = org_prefix


@pytest.mark.level("unit")
def test_reload_fallbacks_prefix_username_false_exact_only():
    """With prefix_username=False and no explicit prefixes, only the bare name is a candidate."""
    from kubetorch.resources.callables.utils import get_names_for_reload_fallbacks

    org = config.username
    try:
        config.set("username", "test-user")
        assert get_names_for_reload_fallbacks("summer", prefix_username=False) == ["summer"]
    finally:
        config.set("username", org)


@pytest.mark.level("unit")
def test_reload_fallbacks_prefix_username_true_includes_prefixed():
    """With prefix_username=True (default), the username-prefixed and bare names are candidates."""
    from kubetorch.resources.callables.utils import get_names_for_reload_fallbacks

    org = config.username
    try:
        config.set("username", "test-user")
        names = get_names_for_reload_fallbacks("summer", prefix_username=True)
        assert "test-user-summer" in names
        assert "summer" in names
    finally:
        config.set("username", org)


@pytest.mark.level("unit")
def test_reload_fallbacks_explicit_prefixes_win_over_prefix_username_false():
    """Explicit reload prefixes take precedence even when prefix_username=False."""
    from kubetorch.resources.callables.utils import get_names_for_reload_fallbacks

    org = config.username
    try:
        config.set("username", "test-user")
        names = get_names_for_reload_fallbacks("summer", prefixes=["prod"], prefix_username=False)
        assert "prod-summer" in names
        assert "summer" not in names
    finally:
        config.set("username", org)


@pytest.mark.level("unit")
def test_get_deployment_mode_exact_match_not_reprefixed(monkeypatch):
    """A service found by its exact (bare) name is returned as-is, not re-prefixed."""
    from kubetorch import cli_utils

    org = config.username
    try:
        config.set("username", "test-user")
        monkeypatch.setattr(
            cli_utils,
            "detect_deployment_mode",
            lambda name, namespace: "Deployment" if name == "train-cnn-r4" else None,
        )
        name, mode = cli_utils.get_deployment_mode("train-cnn-r4", "ns")
        assert name == "train-cnn-r4"
        assert mode == "Deployment"
    finally:
        config.set("username", org)


@pytest.mark.level("unit")
def test_get_deployment_mode_prefixed_fallback(monkeypatch):
    """When the bare name is not found, the username-prefixed name is used."""
    from kubetorch import cli_utils

    org = config.username
    try:
        config.set("username", "test-user")
        monkeypatch.setattr(
            cli_utils,
            "detect_deployment_mode",
            lambda name, namespace: "Deployment" if name == "test-user-summer" else None,
        )
        name, mode = cli_utils.get_deployment_mode("summer", "ns")
        assert name == "test-user-summer"
        assert mode == "Deployment"
    finally:
        config.set("username", org)


@pytest.mark.level("unit")
def test_get_deployment_mode_not_found_raises_exit(monkeypatch):
    """Neither the bare name nor the prefixed name found -> typer.Exit."""
    import typer

    from kubetorch import cli_utils

    org = config.username
    try:
        config.set("username", "test-user")
        monkeypatch.setattr(cli_utils, "detect_deployment_mode", lambda name, namespace: None)
        with pytest.raises(typer.Exit):
            cli_utils.get_deployment_mode("nonexistent", "ns")
    finally:
        config.set("username", org)


@pytest.mark.level("unit")
def test_get_deployment_mode_already_prefixed_name(monkeypatch):
    """A name already starting with the username prefix is returned as-is, not double-prefixed."""
    from kubetorch import cli_utils

    org = config.username
    try:
        config.set("username", "test-user")
        monkeypatch.setattr(
            cli_utils,
            "detect_deployment_mode",
            lambda name, namespace: "Deployment" if name == "test-user-summer" else None,
        )
        name, mode = cli_utils.get_deployment_mode("test-user-summer", "ns")
        assert name == "test-user-summer"
        assert mode == "Deployment"
    finally:
        config.set("username", org)


@pytest.mark.level("unit")
def test_prefix_username_excluded_from_pod_env_vars():
    """prefix_username is client-side only and must not be injected into pod templates."""
    org = config._prefix_username
    try:
        config.set("prefix_username", False)
        assert "KT_PREFIX_USERNAME" not in config._get_config_env_vars()
    finally:
        config._prefix_username = org
