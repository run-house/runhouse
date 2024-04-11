import tempfile
from pathlib import Path

import pytest

import runhouse as rh
from runhouse.servers.http.http_utils import deserialize_data, serialize_data
from runhouse.servers.obj_store import ObjStore


@pytest.mark.servertest
class TestServlet:
    @pytest.mark.level("unit")
    def test_put_resource(self, test_servlet, blob_data):
        with tempfile.TemporaryDirectory() as temp_dir:
            resource_path = Path(temp_dir, "local-blob")
            local_blob = rh.blob(blob_data, path=resource_path)
            resource = local_blob.to(system="file", path=resource_path)

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
            assert deserialize_data(resp.data, resp.serialization).startswith("file_")

    @pytest.mark.level("unit")
    def test_put_obj_local(self, test_servlet, blob_data):
        with tempfile.TemporaryDirectory() as temp_dir:
            resource_path = Path(temp_dir, "local-blob")
            resource = rh.blob(blob_data, path=resource_path)
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
        blob = deserialize_data(resp.data, resp.serialization)
        assert isinstance(blob, rh.Blob)

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
        assert resp.output_type == "result_serialized"
        blob_config = deserialize_data(resp.data, resp.serialization)
        assert isinstance(blob_config, dict)

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
