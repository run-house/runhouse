import time

import pytest
import ray

from runhouse.resources.resource import Resource
from runhouse.servers.http.http_utils import deserialize_data, serialize_data
from runhouse.servers.obj_store import ObjStore

from tests.utils import get_ray_cluster_servlet


@pytest.mark.servertest
class TestServlet:
    @pytest.mark.level("unit")
    def test_put_resource(self, test_servlet):
        resource = Resource(name="local-resource")
        state = {}
        resp = ObjStore.call_actor_method(
            test_servlet,
            "aput_resource_local",
            data=serialize_data(
                (resource.config(condensed=False), state, resource.dryrun), "pickle"
            ),
            serialization="pickle",
        )
        assert resp.output_type == "result_serialized"
        assert deserialize_data(resp.data, resp.serialization) == resource.name

    @pytest.mark.level("unit")
    def test_put_obj_local(self, test_servlet):
        resource = Resource(name="local-resource")
        resp = ObjStore.call_actor_method(
            test_servlet,
            "aput_local",
            key="key1",
            data=serialize_data(resource, "pickle"),
            serialization="pickle",
        )
        assert resp.output_type == "success"

    @pytest.mark.level("unit")
    def test_get_obj(self, test_servlet):
        resp = ObjStore.call_actor_method(
            test_servlet,
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
    def test_get_obj_remote(self, test_servlet):
        resp = ObjStore.call_actor_method(
            test_servlet,
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
    def test_get_obj_does_not_exist(self, test_servlet):
        resp = ObjStore.call_actor_method(
            test_servlet,
            "aget_local",
            key="abcdefg",
            default=KeyError,
            serialization="pickle",
            remote=False,
        )
        assert resp.output_type == "exception"
        error = deserialize_data(resp.data["error"], "pickle")
        assert isinstance(error, KeyError)

    @pytest.mark.level("local")
    def test_failed_cluster_servlet(self):

        invalid_cluster_config = {
            "api_server_url": "https://api.run.house.invalid",
            "name": "mocked-cluster",
        }
        cluster_servlet = get_ray_cluster_servlet(
            cluster_config=invalid_cluster_config, name="invalid_cluster_servlet"
        )
        try:
            # waiting for a seconds so the async cluster status check thread will be executed and then fail and kill the clsuter servlet.
            time.sleep(1)
            cluster_servlet = ray.get_actor(
                name="invalid_cluster_servlet", namespace="runhouse"
            )
            # if the cluster servlet is not deleted -> it is not none -> not cluster_servlet = False ->
            # the test fails, because we expect that the cluster servlet will be deleted
            assert not cluster_servlet

        except ValueError as e:
            assert (
                str(e)
                == "Failed to look up actor with name 'invalid_cluster_servlet'. This could because 1. You are trying to look up a named actor you didn't create. 2. The named actor died. 3. You did not use a namespace matching the namespace of the actor."
            )
