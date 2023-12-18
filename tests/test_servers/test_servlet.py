import tempfile
from pathlib import Path

import pytest

import runhouse as rh
from runhouse.servers.http.auth import hash_token
from runhouse.servers.http.http_server import HTTPServer
from runhouse.servers.http.http_utils import b64_unpickle, Message, pickle_b64

from tests.test_servers.conftest import summer
from tests.utils import test_account


class TestServlet:
    @pytest.mark.level("unit")
    def test_put_resource(self, base_servlet, blob_data):
        with tempfile.TemporaryDirectory() as temp_dir:
            resource_path = Path(temp_dir, "local-blob")
            local_blob = rh.blob(blob_data, path=resource_path)
            resource = local_blob.to(system="file", path=resource_path)

            state = {}
            message = Message(
                data=pickle_b64((resource.config_for_rns, state, resource.dryrun)),
            )
            resp = HTTPServer.call_servlet_method(
                base_servlet, "put_resource", [message]
            )

            assert resp.output_type == "result"
            assert b64_unpickle(resp.data).startswith("file_")

    @pytest.mark.level("unit")
    def test_put_obj(self, base_servlet, blob_data):
        with tempfile.TemporaryDirectory() as temp_dir:
            resource_path = Path(temp_dir, "local-blob")
            resource = rh.blob(blob_data, path=resource_path)
            message = Message(data=pickle_b64(resource), key="key1")
            resp = HTTPServer.call_servlet_method(
                base_servlet, "put_object", [message.key, message.data]
            )
            assert resp.output_type == "success"

    @pytest.mark.level("unit")
    def test_get_obj(self, base_servlet):
        remote = False
        stream = True
        resp = HTTPServer.call_servlet_method(
            base_servlet,
            "get",
            ["key1", remote, stream],
        )
        assert resp.output_type == "result"
        blob = b64_unpickle(resp.data)
        assert isinstance(blob, rh.Blob)

    @pytest.mark.level("unit")
    def test_get_obj_config(self, base_servlet):
        remote = True
        stream = True
        resp = HTTPServer.call_servlet_method(
            base_servlet,
            "get",
            ["key1", remote, stream],
        )
        assert resp.output_type == "config"
        blob_config = resp.data
        assert isinstance(blob_config, dict)

    @pytest.mark.level("unit")
    def test_get_obj_as_ref(self, base_servlet):
        import ray

        remote = False
        stream = True
        resp = HTTPServer.call_servlet_method(
            base_servlet, "get", ["key1", remote, stream], block=False
        )
        assert isinstance(resp, ray.ObjectRef)

        resp_object = ray.get(resp)
        assert isinstance(b64_unpickle(resp_object.data), rh.Blob)

    @pytest.mark.level("unit")
    def test_get_obj_ref_as_config(self, base_servlet):
        import ray

        remote = True
        stream = True
        resp = HTTPServer.call_servlet_method(
            base_servlet, "get", ["key1", remote, stream], block=False
        )
        assert isinstance(resp, ray.ObjectRef)

        resp_object = ray.get(resp)
        assert resp_object.output_type == "config"
        assert isinstance(resp_object.data, dict)

    @pytest.mark.level("unit")
    def test_get_obj_does_not_exist(self, base_servlet):
        remote = False
        stream = True
        resp = HTTPServer.call_servlet_method(
            base_servlet,
            "get",
            ["abcdefg", remote, stream],
        )
        assert resp.output_type == "exception"
        assert isinstance(b64_unpickle(resp.error), KeyError)

    @pytest.mark.level("unit")
    def test_get_keys(self, base_servlet):
        resp = HTTPServer.call_servlet_method(base_servlet, "get_keys", [])
        assert resp.output_type == "result"
        keys: list = b64_unpickle(resp.data)
        assert "key1" in keys

    @pytest.mark.level("unit")
    def test_rename_object(self, base_servlet):
        message = Message(data=pickle_b64(("key1", "key2")))
        resp = HTTPServer.call_servlet_method(base_servlet, "rename_object", [message])
        assert resp.output_type == "success"

        resp = HTTPServer.call_servlet_method(base_servlet, "get_keys", [])
        assert resp.output_type == "result"

        keys: list = b64_unpickle(resp.data)
        assert "key2" in keys

    @pytest.mark.level("unit")
    def test_delete_obj(self, base_servlet):
        remote = False
        keys = ["key2"]
        message = Message(data=pickle_b64((keys)))
        resp = HTTPServer.call_servlet_method(base_servlet, "delete_obj", [message])
        assert resp.output_type == "result"
        assert b64_unpickle(resp.data) == ["key2"]

        resp = HTTPServer.call_servlet_method(
            base_servlet,
            "get",
            ["key2", remote, True],
        )
        assert resp.output_type == "exception"
        assert isinstance(b64_unpickle(resp.error), KeyError)

    @pytest.mark.skip("Not implemented yet.")
    @pytest.mark.level("unit")
    def test_call(self, base_servlet, local_docker_cluster_public_key_logged_in):
        token_hash = None
        den_auth = False
        remote_func = rh.function(
            summer, system=local_docker_cluster_public_key_logged_in
        )

        method_name = "call"
        module_name = remote_func.name
        args = (1, 2)
        kwargs = {}
        serialization = "none"
        resp = HTTPServer.call_servlet_method(
            base_servlet,
            "call",
            [
                module_name,
                method_name,
                args,
                kwargs,
                serialization,
                token_hash,
                den_auth,
            ],
        )

        assert b64_unpickle(resp.data) == 3

    @pytest.mark.skip("Not implemented yet.")
    @pytest.mark.level("unit")
    def test_call_with_den_auth(self, base_servlet):
        with test_account() as test_account_dict:
            token_hash = hash_token(test_account_dict["token"])
            den_auth = True
            remote_func = rh.function(summer).save()

        method_name = "call"
        module_name = remote_func.name
        args = (1, 2)
        kwargs = {}
        serialization = "none"

        resp = HTTPServer.call_servlet_method(
            base_servlet,
            "call",
            [
                module_name,
                method_name,
                args,
                kwargs,
                serialization,
                token_hash,
                den_auth,
            ],
        )

        assert b64_unpickle(resp.data) == 3

    @pytest.mark.skip("Not implemented yet.")
    @pytest.mark.level("unit")
    def test_call_module_method_(self, base_servlet):
        with test_account():
            token_hash = None
            den_auth = False
            remote_func = rh.function(summer).save()

        method_name = "call"
        module_name = remote_func.name
        args = (1, 2)
        kwargs = {}
        message = Message(data=pickle_b64(args, kwargs))

        resp = HTTPServer.call_servlet_method(
            base_servlet,
            "call_module_method",
            [module_name, method_name, message, token_hash, den_auth],
        )

        assert b64_unpickle(resp.data) == 3

    @pytest.mark.skip("Not implemented yet.")
    @pytest.mark.level("unit")
    def test_call_module_method_with_den_auth(self, base_servlet):
        with test_account() as test_account_dict:
            token_hash = hash_token(test_account_dict["token"])
            den_auth = True
            remote_func = rh.function(summer).save()

        method_name = "call"
        module_name = remote_func.name
        args = (1, 2)
        kwargs = {}
        message = Message(data=pickle_b64(args, kwargs))

        resp = HTTPServer.call_servlet_method(
            base_servlet,
            "call_module_method",
            [module_name, method_name, message, token_hash, den_auth],
        )

        assert b64_unpickle(resp.data) == 3

    @pytest.mark.skip("Not implemented yet.")
    @pytest.mark.level("unit")
    def cancel_run(self, base_servlet):
        pass
