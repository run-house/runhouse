import pytest

from runhouse.resources.resource import Resource
from runhouse.servers.http.http_utils import deserialize_data, serialize_data
from runhouse.servers.obj_store import ObjStore


@pytest.mark.servertest
class TestServlet:
    @pytest.mark.level("unit")
    def test_put_resource(self, test_env_servlet):
        resource = Resource(name="local-resource")
        state = {}
        resp = ObjStore.call_actor_method(
            test_env_servlet,
            "aput_resource_local",
            data=serialize_data(
                (resource.config(condensed=False), state, resource.dryrun), "pickle"
            ),
            serialization="pickle",
        )
        assert resp.output_type == "result_serialized"
        assert deserialize_data(resp.data, resp.serialization) == resource.name

    @pytest.mark.level("unit")
    def test_put_obj_local(self, test_env_servlet):
        resource = Resource(name="local-resource")
        resp = ObjStore.call_actor_method(
            test_env_servlet,
            "aput_local",
            key="key1",
            data=serialize_data(resource, "pickle"),
            serialization="pickle",
        )
        assert resp.output_type == "success"

    @pytest.mark.level("unit")
    def test_get_obj(self, test_env_servlet):
        resp = ObjStore.call_actor_method(
            test_env_servlet,
            "aget_local",
            key="key1",
            default=KeyError,
            serialization="pickle",
            remote=False,
        )
        assert resp.output_type == "result_serialized"
        resource = deserialize_data(resp.data, resp.serialization)
        assert isinstance(resource, Resource)

    @pytest.mark.level("unit")
    def test_get_obj_remote(self, test_env_servlet):
        resp = ObjStore.call_actor_method(
            test_env_servlet,
            "aget_local",
            key="key1",
            default=KeyError,
            serialization="pickle",
            remote=True,
        )
        assert resp.output_type == "config"
        resource_config = deserialize_data(resp.data, resp.serialization)
        assert isinstance(resource_config, dict)

    @pytest.mark.level("unit")
    def test_get_obj_does_not_exist(self, test_env_servlet):
        resp = ObjStore.call_actor_method(
            test_env_servlet,
            "aget_local",
            key="abcdefg",
            default=KeyError,
            serialization="pickle",
            remote=False,
        )
        assert resp.output_type == "exception"
        error = deserialize_data(resp.data["error"], "pickle")
        assert isinstance(error, KeyError)
