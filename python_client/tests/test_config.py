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
