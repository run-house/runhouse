import tempfile
from pathlib import Path

import pytest

import runhouse as rh
from runhouse.servers.http.http_utils import b64_unpickle, pickle_b64
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
                "put_resource_local",
                data=pickle_b64((resource.config_for_rns, state, resource.dryrun)),
                serialization="pickle",
            )

            assert resp.output_type == "result_serialized"
            assert b64_unpickle(resp.data).startswith("file_")

    @pytest.mark.level("unit")
    def test_put_obj_local(self, test_servlet, blob_data):
        with tempfile.TemporaryDirectory() as temp_dir:
            resource_path = Path(temp_dir, "local-blob")
            resource = rh.blob(blob_data, path=resource_path)
            resp = ObjStore.call_actor_method(
                test_servlet,
                "put_local",
                key="key1",
                data=pickle_b64(resource),
                serialization="pickle",
            )
            assert resp.output_type == "success"

    @pytest.mark.level("unit")
    def test_get_obj(self, test_servlet):
        resp = ObjStore.call_actor_method(
            test_servlet,
            "get_local",
            key="key1",
            default=KeyError,
            serialization="pickle",
            remote=False,
        )
        assert resp.output_type == "result_serialized"
        blob = b64_unpickle(resp.data)
        assert isinstance(blob, rh.Blob)

    @pytest.mark.level("unit")
    def test_get_obj_remote(self, test_servlet):
        resp = ObjStore.call_actor_method(
            test_servlet,
            "get_local",
            key="key1",
            default=KeyError,
            serialization="pickle",
            remote=True,
        )
        assert resp.output_type == "result_serialized"
        blob_config = b64_unpickle(resp.data)
        assert isinstance(blob_config, dict)

    @pytest.mark.level("unit")
    def test_get_obj_does_not_exist(self, test_servlet):
        resp = ObjStore.call_actor_method(
            test_servlet,
            "get_local",
            key="abcdefg",
            default=KeyError,
            serialization="pickle",
            remote=False,
        )
        assert resp.output_type == "exception"
        assert isinstance(b64_unpickle(resp.error), KeyError)
