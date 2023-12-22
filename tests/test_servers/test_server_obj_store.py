import pytest

from runhouse.servers.http.auth import hash_token, update_cache_for_user

from tests.test_servers.conftest import BASE_ENV_ACTOR_NAME, CACHE_ENV_ACTOR_NAME
from tests.utils import test_account


@pytest.mark.parametrize("obj_store", [BASE_ENV_ACTOR_NAME], indirect=True)
class TestBaseEnvObjStore:
    """Start object store in a local base env servlet"""

    @pytest.mark.level("unit")
    def test_put_and_get_string(self, obj_store):
        key = "k1"
        value = "v1"
        obj_store.put(key, value)
        res = obj_store.get(key)
        assert res == value

    @pytest.mark.level("unit")
    def test_put_and_get_numeric(self, obj_store):
        key = "numeric_key"
        value = 12345
        obj_store.put(key, value)
        res = obj_store.get(key)
        assert res == value

    @pytest.mark.level("unit")
    def test_put_and_get_list(self, obj_store):
        key = "list_key"
        value = [1, 2, 3, 4, 5]
        obj_store.put(key, value)
        res = obj_store.get(key)
        assert res == value

    @pytest.mark.level("unit")
    def test_put_and_get_none(self, obj_store):
        key = "none_key"
        value = None
        obj_store.put(key, value)
        res = obj_store.get(key)
        assert res == value

    @pytest.mark.level("unit")
    def test_put_and_get_binary(self, obj_store):
        key = "binary_key"
        value = b"\x00\x01\x02"
        obj_store.put(key, value)
        res = obj_store.get(key)
        assert res == value

    @pytest.mark.level("unit")
    def test_put_and_get_custom_object(self, obj_store, local_blob):
        from runhouse import Blob

        key = "custom_object_key"
        value = local_blob
        obj_store.put(key, value)
        res = obj_store.get(key)

        assert isinstance(res, Blob) and res.data == value.data

    @pytest.mark.level("unit")
    def test_put_and_get_with_non_string_key(self, obj_store):
        key = 123
        value = "value"
        obj_store.put(key, value)
        res = obj_store.get(key)
        assert res == value

    @pytest.mark.level("unit")
    def test_get_nonexistent_key(self, obj_store):
        key = "nonexistent_key"
        default_value = None
        res = obj_store.get(key, default=default_value)
        assert res == default_value

    @pytest.mark.level("unit")
    def test_list_keys(self, obj_store):
        keys = obj_store.keys()
        assert isinstance(keys, list)
        assert "k1" in keys

    @pytest.mark.level("unit")
    def test_delete_key(self, obj_store):
        key = "k1"
        obj_store.delete(key)
        res = obj_store.get(key, default=None)
        assert res is None

    @pytest.mark.level("unit")
    def test_pop(self, obj_store):
        key = "new_key"
        obj_store.put(key, "v1")
        value = obj_store.pop(key)
        assert value == "v1"

        res = obj_store.get(key, default=None)
        assert res is None

    @pytest.mark.level("unit")
    def test_get_list(self, obj_store):
        keys = ["k1", "k2", "k3"]
        vals = ["v1", "v2", "v3"]

        for k, v in zip(keys, vals):
            obj_store.put(k, v)

        res = obj_store.get_list(keys)
        assert res == ["v1", "v2", "v3"]

    @pytest.mark.level("unit")
    def test_rename(self, obj_store):
        key = "k1"
        new_key = "k2"
        obj_store.rename(key, new_key)
        res = obj_store.get(new_key)
        assert res == "v1"

    @pytest.mark.level("unit")
    def test_get_env(self, obj_store):
        env = obj_store.get_env("k2")
        assert env == "base"

    @pytest.mark.level("unit")
    def test_put_and_get_new_env(self, obj_store):
        from runhouse import env, Env

        key = "new_env"
        new_env = env(
            reqs=["pytest"],
            working_dir=None,
            name="new_env",
        )
        obj_store.put_env(key, new_env)
        res = obj_store.get_env(key)
        assert isinstance(res, Env) and res.name == new_env.name

    @pytest.mark.level("unit")
    def test_put_and_get_obj_ref(self, obj_store):
        key = "obj_ref"
        obj_store.put_obj_ref(key, "new_obj_ref")
        res = obj_store.get_obj_ref(key)
        assert res == "new_obj_ref"

    @pytest.mark.level("unit")
    def test_contains(self, obj_store):
        key = "obj_ref_random"
        res = obj_store.contains(key)
        assert res is False

        obj_store.put_obj_ref(key, "new_obj_ref")
        res = obj_store.contains(key)
        assert res is True

    @pytest.mark.level("unit")
    def test_pop_env(self, obj_store):
        env = "new_env"
        obj_store.pop_env(env)
        res = obj_store.get_env(env)
        assert res is None

    @pytest.mark.level("unit")
    def test_clear_env(self, obj_store):
        obj_store.clear_env()
        res = obj_store.get_env("new_env")
        assert res is None

    @pytest.mark.level("unit")
    def test_clear(self, obj_store):
        obj_store.clear()
        res = obj_store.get_env("new_env")
        assert res is None

    @pytest.mark.skip("Not implemented yet.")
    @pytest.mark.level("unit")
    def test_cancel(self, obj_store):
        key = "obj_ref"
        obj_store.put_obj_ref(key, "new_obj_ref")
        obj_store.cancel(key)

        obj_ref = obj_store.get_obj_ref(key, default=None)
        assert obj_ref is None

    @pytest.mark.skip("Not implemented yet.")
    @pytest.mark.level("unit")
    def test_cancel_all(self, obj_store):
        obj_store.cancel_all()
        assert obj_store.keys() == []


@pytest.mark.parametrize("obj_store", [CACHE_ENV_ACTOR_NAME], indirect=True)
class TestAuthCacheObjStore:
    """Start object store in a local auth cache servlet"""

    @pytest.mark.level("unit")
    def test_save_resources_to_obj_store_cache(self, obj_store):
        with test_account() as test_account_dict:
            token = test_account_dict["token"]
            hashed_token = hash_token(token)

            # Add test account resources to the local cache
            update_cache_for_user(token)
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
        with test_account() as test_account_dict:
            token = "abc"
            resource_uri = f"/{test_account_dict['username']}/summer"
            access_level = obj_store.resource_access_level(
                hash_token(token), resource_uri
            )
            assert access_level is None
