import pytest

from runhouse.servers.http.auth import hash_token

from tests.test_servers.conftest import BASE_ENV_ACTOR_NAME
from tests.utils import friend_account, get_test_obj_store


@pytest.mark.servertest
@pytest.mark.parametrize("obj_store", [BASE_ENV_ACTOR_NAME], indirect=True)
class TestBaseEnvObjStore:
    """Start object store in a local base env servlet"""

    @pytest.mark.level("unit")
    @pytest.mark.parametrize("key", ["k1", 123])
    @pytest.mark.parametrize(
        "value",
        ["test", 1, 1.0, True, False, None, b"\x00\x01\x02", [1, 2, 3], ["1", 1, 1.5]],
    )
    def test_put_get_delete_string(self, obj_store, key, value):
        assert obj_store.keys() == []

        key = "k1"
        obj_store.put(key, value)
        assert obj_store.keys() == [key]

        res = obj_store.get(key)
        assert res == value
        assert obj_store.get_env_servlet_name_for_key(key) == obj_store.servlet_name

        obj_store.put(key, "overwrite_value")
        assert obj_store.keys() == [key]

        res = obj_store.get(key)
        assert res == "overwrite_value"
        assert obj_store.get_env_servlet_name_for_key(key) == obj_store.servlet_name

        obj_store.delete(key)
        assert obj_store.keys() == []
        assert obj_store.get(key, default=None) is None
        assert obj_store.get_env_servlet_name_for_key(key) is None

    @pytest.mark.level("unit")
    @pytest.mark.parametrize("default_value", [1, None, "asdf"])
    def test_get_default_behavior(self, obj_store, default_value):
        assert obj_store.keys() == []

        assert obj_store.get("a", default=default_value) == default_value
        assert obj_store.get("b", default=default_value) == default_value

    @pytest.mark.level("unit")
    def test_get_default_key_error(self, obj_store):
        assert obj_store.keys() == []

        try:
            obj_store.get("c", default=KeyError)
        except KeyError:
            return

        assert False, "Should have raised KeyError"

    @pytest.mark.level("unit")
    def test_pop(self, obj_store):
        assert obj_store.keys() == []

        key = "new_key"
        obj_store.put(key, "v1")
        value = obj_store.pop(key)
        assert value == "v1"

        res = obj_store.get(key, default=None)
        assert res is None

    @pytest.mark.level("unit")
    def test_get_list(self, obj_store):
        assert obj_store.keys() == []

        keys = ["k1", "k2", "k3"]
        vals = ["v1", "v2", "v3"]

        for k, v in zip(keys, vals):
            obj_store.put(k, v)

        res = obj_store.get_list(keys)
        assert res == ["v1", "v2", "v3"]

    @pytest.mark.level("unit")
    def test_rename(self, obj_store):
        assert obj_store.keys() == []

        key = "k1"
        new_key = "k2"
        val = "v1"

        obj_store.put(key, val)
        obj_store.rename(key, new_key)
        res = obj_store.get(new_key)
        assert res == val
        assert obj_store.get(key, default=None) is None
        assert obj_store.keys() == [new_key]
        assert obj_store.get_env_servlet_name_for_key(new_key) == obj_store.servlet_name
        assert obj_store.get_env_servlet_name_for_key(key) is None

    @pytest.mark.level("unit")
    def test_clear(self, obj_store):
        assert obj_store.keys() == []

        keys = ["k1", "k2", "k3"]
        vals = ["v1", "v2", "v3"]

        for k, v in zip(keys, vals):
            obj_store.put(k, v)

        assert obj_store.keys() == keys

        obj_store.clear()
        assert obj_store.keys() == []

    @pytest.mark.level("unit")
    def test_many_env_servlets(self, obj_store):
        assert obj_store.keys() == []

        other_obj_store = get_test_obj_store("other")
        assert other_obj_store.keys() == []

        obj_store.put("k1", "v1")
        other_obj_store.put("k2", "v2")
        other_obj_store.put("k3", "v3")

        assert obj_store.keys() == ["k1", "k2", "k3"]
        assert other_obj_store.keys() == ["k1", "k2", "k3"]

        assert obj_store.get("k1") == "v1"
        assert obj_store.get("k2") == "v2"
        assert obj_store.get("k3") == "v3"
        assert other_obj_store.get("k1") == "v1"
        assert other_obj_store.get("k2") == "v2"
        assert other_obj_store.get("k3") == "v3"

        assert obj_store.get_env_servlet_name_for_key("k1") == obj_store.servlet_name
        assert (
            obj_store.get_env_servlet_name_for_key("k2") == other_obj_store.servlet_name
        )
        assert (
            obj_store.get_env_servlet_name_for_key("k3") == other_obj_store.servlet_name
        )
        assert (
            other_obj_store.get_env_servlet_name_for_key("k1") == obj_store.servlet_name
        )
        assert (
            other_obj_store.get_env_servlet_name_for_key("k2")
            == other_obj_store.servlet_name
        )
        assert (
            other_obj_store.get_env_servlet_name_for_key("k3")
            == other_obj_store.servlet_name
        )

        # Technically, "k1" is only present on the base env servlet,
        # and "k2" and "k3" are only present on the other env servlet
        # These methods are static, we can run them from either store
        assert obj_store.keys_for_env_servlet_name(obj_store.servlet_name) == ["k1"]
        assert obj_store.get_from_env_servlet_name(obj_store.servlet_name, "k1") == "v1"
        assert obj_store.get_from_env_servlet_name(obj_store.servlet_name, "k2") is None
        assert obj_store.get_from_env_servlet_name(obj_store.servlet_name, "k3") is None

        assert obj_store.keys_for_env_servlet_name(other_obj_store.servlet_name) == [
            "k2",
            "k3",
        ]
        assert (
            obj_store.get_from_env_servlet_name(other_obj_store.servlet_name, "k1")
            is None
        )
        assert (
            obj_store.get_from_env_servlet_name(other_obj_store.servlet_name, "k2")
            == "v2"
        )
        assert (
            obj_store.get_from_env_servlet_name(other_obj_store.servlet_name, "k3")
            == "v3"
        )

        # Overwriting "k2" from obj_store instead of other_obj_store
        obj_store.put("k2", "changed")
        assert obj_store.keys() == ["k1", "k3", "k2"]
        assert obj_store.get("k2") == "changed"
        assert other_obj_store.get("k2") == "changed"
        assert obj_store.get_env_servlet_name_for_key("k2") == obj_store.servlet_name
        assert (
            other_obj_store.get_env_servlet_name_for_key("k2") == obj_store.servlet_name
        )

        assert obj_store.keys_for_env_servlet_name(obj_store.servlet_name) == [
            "k1",
            "k2",
        ]
        assert obj_store.get_from_env_servlet_name(obj_store.servlet_name, "k1") == "v1"
        assert (
            obj_store.get_from_env_servlet_name(obj_store.servlet_name, "k2")
            == "changed"
        )
        assert obj_store.get_from_env_servlet_name(obj_store.servlet_name, "k3") is None

        assert obj_store.keys_for_env_servlet_name(other_obj_store.servlet_name) == [
            "k3"
        ]
        assert (
            obj_store.get_from_env_servlet_name(other_obj_store.servlet_name, "k1")
            is None
        )
        assert (
            obj_store.get_from_env_servlet_name(other_obj_store.servlet_name, "k2")
            is None
        )
        assert (
            obj_store.get_from_env_servlet_name(other_obj_store.servlet_name, "k3")
            == "v3"
        )

        # Renaming "k2" to "key_changed" from other_obj_store
        # Even though "k2" is technically on base object store.
        other_obj_store.rename("k2", "key_changed")
        assert obj_store.keys() == ["k1", "k3", "key_changed"]
        assert obj_store.get("key_changed") == "changed"
        assert other_obj_store.get("key_changed") == "changed"
        assert (
            obj_store.get_env_servlet_name_for_key("key_changed")
            == obj_store.servlet_name
        )
        assert (
            other_obj_store.get_env_servlet_name_for_key("key_changed")
            == obj_store.servlet_name
        )

        assert obj_store.keys_for_env_servlet_name(obj_store.servlet_name) == [
            "k1",
            "key_changed",
        ]
        assert obj_store.get_from_env_servlet_name(obj_store.servlet_name, "k1") == "v1"
        assert (
            obj_store.get_from_env_servlet_name(obj_store.servlet_name, "key_changed")
            == "changed"
        )
        assert obj_store.get_from_env_servlet_name(obj_store.servlet_name, "k3") is None

        assert obj_store.keys_for_env_servlet_name(other_obj_store.servlet_name) == [
            "k3"
        ]
        assert (
            obj_store.get_from_env_servlet_name(other_obj_store.servlet_name, "k1")
            is None
        )
        assert (
            obj_store.get_from_env_servlet_name(
                other_obj_store.servlet_name, "key_changed"
            )
            is None
        )
        assert (
            obj_store.get_from_env_servlet_name(other_obj_store.servlet_name, "k3")
            == "v3"
        )

        # Renaming "key_changed" to "k3"
        obj_store.rename("key_changed", "k3")
        assert obj_store.keys() == ["k1", "k3"]
        assert obj_store.get("k3") == "changed"
        assert other_obj_store.get("k3") == "changed"
        assert obj_store.get_env_servlet_name_for_key("k3") == obj_store.servlet_name
        assert (
            other_obj_store.get_env_servlet_name_for_key("k3") == obj_store.servlet_name
        )

        assert obj_store.keys_for_env_servlet_name(obj_store.servlet_name) == [
            "k1",
            "k3",
        ]
        assert obj_store.get_from_env_servlet_name(obj_store.servlet_name, "k1") == "v1"
        assert (
            obj_store.get_from_env_servlet_name(obj_store.servlet_name, "k3")
            == "changed"
        )
        assert (
            obj_store.get_from_env_servlet_name(obj_store.servlet_name, "key_changed")
            is None
        )

        assert obj_store.keys_for_env_servlet_name(other_obj_store.servlet_name) == []
        assert (
            obj_store.get_from_env_servlet_name(other_obj_store.servlet_name, "k1")
            is None
        )
        assert (
            obj_store.get_from_env_servlet_name(other_obj_store.servlet_name, "k3")
            is None
        )
        assert (
            obj_store.get_from_env_servlet_name(
                other_obj_store.servlet_name, "key_changed"
            )
            is None
        )

        # Popping "k3" from other_obj_store
        res = other_obj_store.pop("k3")
        assert res == "changed"
        assert obj_store.keys() == ["k1"]
        assert obj_store.get("k3") is None
        assert other_obj_store.get("k3") is None
        assert obj_store.get_env_servlet_name_for_key("k3") is None
        assert other_obj_store.get_env_servlet_name_for_key("k3") is None

        assert obj_store.keys_for_env_servlet_name(obj_store.servlet_name) == ["k1"]
        assert obj_store.get_from_env_servlet_name(obj_store.servlet_name, "k1") == "v1"
        assert obj_store.keys_for_env_servlet_name(other_obj_store.servlet_name) == []


@pytest.mark.servertest
@pytest.mark.parametrize("obj_store", [BASE_ENV_ACTOR_NAME], indirect=True)
class TestAuthCacheObjStore:
    """Start object store in a local auth cache servlet"""

    @pytest.mark.level("unit")
    def test_save_resources_to_obj_store_cache(self, obj_store):
        with friend_account() as test_account_dict:
            token = test_account_dict["token"]
            hashed_token = hash_token(token)

            # Add test account resources to the local cache
            obj_store.add_user_to_auth_cache(token)
            resources = obj_store.user_resources(hashed_token)
            assert resources

            resource_uri = f"/{test_account_dict['username']}/summer"
            access_level = obj_store.resource_access_level(hashed_token, resource_uri)

            assert access_level == "write"

    @pytest.mark.level("unit")
    def test_no_resources_for_invalid_token(self, obj_store):
        token = "abc"
        resources = obj_store.user_resources(hash_token(token))
        assert not resources

    @pytest.mark.level("unit")
    def test_no_resource_access_for_invalid_token(self, obj_store):
        with friend_account() as test_account_dict:
            token = "abc"
            resource_uri = f"/{test_account_dict['username']}/summer"
            access_level = obj_store.resource_access_level(
                hash_token(token), resource_uri
            )
            assert access_level is None
