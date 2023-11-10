import os
import shutil
import unittest
from pathlib import Path

import runhouse as rh
from runhouse.servers.http.auth import hash_token
from runhouse.servers.http.http_server import HTTPServer
from runhouse.servers.http.http_utils import b64_unpickle, Message, pickle_b64

from tests.test_servers.conftest import summer


class TestServlet:
    def test_put_resource(self, base_servlet, local_blob):
        resource_path = Path("~/rh/blob/local-blob").expanduser()
        resource_dir = resource_path.parent
        try:
            state = {}
            resource = local_blob.to(system="file", path=resource_path)
            message = Message(
                data=pickle_b64((resource.config_for_rns, state, resource.dryrun)),
                env="base_env",
            )
            resp = HTTPServer.call_servlet_method(
                base_servlet, "put_resource", [message]
            )

            assert resp.output_type == "result"
            assert b64_unpickle(resp.data).startswith("file_")

        finally:
            if os.path.exists(resource_path):
                shutil.rmtree(resource_dir)

    def test_put_obj(self, base_servlet, blob_data):
        resource_path = Path("~/rh/blob/local-blob").expanduser()
        resource_dir = resource_path.parent
        try:
            resource = rh.blob(blob_data, path=resource_path)
            message = Message(data=pickle_b64(resource), key="key1")
            resp = HTTPServer.call_servlet_method(
                base_servlet, "put_object", [message.key, message.data]
            )
            assert resp.output_type == "success"
        finally:
            if os.path.exists(resource_path):
                shutil.rmtree(resource_dir)

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

    def test_get_keys(self, base_servlet):
        resp = HTTPServer.call_servlet_method(base_servlet, "get_keys", [])
        assert resp.output_type == "result"
        keys: list = b64_unpickle(resp.data)
        assert "key1" in keys

    def test_rename_object(self, base_servlet):
        message = Message(data=pickle_b64(("key1", "key2")))
        resp = HTTPServer.call_servlet_method(base_servlet, "rename_object", [message])
        assert resp.output_type == "success"

        resp = HTTPServer.call_servlet_method(base_servlet, "get_keys", [])
        assert resp.output_type == "result"

        keys: list = b64_unpickle(resp.data)
        assert "key2" in keys

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

    def test_add_secrets(self, base_servlet):
        secrets = {"aws": {"access_key": "abc123", "secret_key": "abc123"}}
        message = Message(data=pickle_b64(secrets))
        resp = HTTPServer.call_servlet_method(base_servlet, "add_secrets", [message])

        assert resp.output_type == "result"
        assert not b64_unpickle(resp.data)

    def test_add_secrets_for_unsupported_provider(self, base_servlet):
        secrets = {"test_provider": {"access_key": "abc123"}}
        message = Message(data=pickle_b64(secrets))
        resp = HTTPServer.call_servlet_method(base_servlet, "add_secrets", [message])
        assert resp.output_type == "result"

        resp_data = b64_unpickle(resp.data)
        assert isinstance(resp_data, dict)
        assert "test_provider is not a Runhouse builtin provider" in resp_data.values()

    @unittest.skip("Not implemented yet.")
    def test_call(self, base_servlet, test_account, base_cluster):
        token_hash = None
        den_auth = False
        remote_func = rh.function(summer, system=base_cluster)

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

    @unittest.skip("Not implemented yet.")
    def test_call_with_den_auth(self, base_servlet, test_account):
        with test_account as t:
            token_hash = hash_token(t["test_token"])
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

    @unittest.skip("Not implemented yet.")
    def test_call_module_method_(self, base_servlet, test_account):
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

    @unittest.skip("Not implemented yet.")
    def test_call_module_method_with_den_auth(self, base_servlet, test_account):
        with test_account as t:
            token_hash = hash_token(t["test_token"])
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

    @unittest.skip("Not implemented yet.")
    def cancel_run(self, base_servlet):
        pass


if __name__ == "__main__":
    unittest.main()
