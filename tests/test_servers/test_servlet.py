import logging
import time

import pytest

from runhouse.resources.resource import Resource
from runhouse.servers.http.http_utils import deserialize_data, serialize_data
from runhouse.servers.obj_store import ObjStore

from tests.utils import init_remote_cluster_servlet_actor


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
        import ray

        # need to initialize ray so the cluster servlet will be initialized when calling get_cluster_servlet.
        ray.init(
            ignore_reinit_error=True,
            logging_level=logging.ERROR,
            namespace="runhouse",
        )

        current_ip = ray.get_runtime_context().worker.node_ip_address

        cluster_config = {"name": "mocked_cluster"}

        init_remote_cluster_servlet_actor(
            current_ip=current_ip,
            servlet_name="invalid_cluster_servlet",
            cluster_config=cluster_config,
            runtime_env={
                "env_vars": {
                    "API_SERVER_URL": "https://api.run.house.invalid",
                    "RH_LOG_LEVEL": "DEBUG",
                }
            },
        )

        # wait for the async cluster status check thread to fail and kill the cluster servlet.
        time.sleep(5)

        with pytest.raises(ValueError) as error:

            ray.get_actor(name="invalid_cluster_servlet", namespace="runhouse")
        assert (
            str(error.value)
            == "Failed to look up actor with name 'invalid_cluster_servlet'. This could because 1. You are trying to look up a named actor you didn't create. 2. The named actor died. 3. You did not use a namespace matching the namespace of the actor."
        )
