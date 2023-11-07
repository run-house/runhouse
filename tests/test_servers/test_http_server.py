import ipaddress
import unittest

from pathlib import Path
from unittest.mock import patch

import pytest

import runhouse as rh

from cryptography import x509
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives.serialization import load_pem_private_key
from cryptography.x509 import load_pem_x509_certificate
from cryptography.x509.oid import NameOID

from runhouse.rns.utils.api import resolve_absolute_path
from runhouse.servers.http.certs import TLSCertConfig
from runhouse.servers.http.http_utils import b64_unpickle, pickle_b64

from tests.test_servers.conftest import summer


@pytest.mark.usefixtures("docker_container")
class TestHTTPServer:
    def test_get_cert(self, http_client):
        response = http_client.get("/cert")
        assert response.status_code == 200

        error_b64 = response.json().get("error")
        error_message = b64_unpickle(error_b64)

        assert isinstance(error_message, FileNotFoundError)
        assert "No certificate found on cluster in path" in str(error_message)

    def test_check_server(self, http_client):
        response = http_client.get("/check")
        assert response.status_code == 200

    def test_put_resource(self, http_client, local_blob, base_cluster):
        state = None
        resource = local_blob.to(base_cluster)
        data = pickle_b64((resource.config_for_rns, state, resource.dryrun))
        response = http_client.post("/resource", json={"data": data})
        assert response.status_code == 200

    def test_put_object(self, http_client):
        test_list = list(range(5, 50, 2)) + ["a string"]
        response = http_client.post(
            "/object", json={"data": pickle_b64(test_list), "key": "key1"}
        )
        assert response.status_code == 200

    def test_rename_object(self, http_client):
        old_key = "key1"
        new_key = "key2"
        data = pickle_b64((old_key, new_key))
        response = http_client.put("/object", json={"data": data})
        assert response.status_code == 200

    def test_get_keys(self, http_client):
        response = http_client.get("/keys")
        assert response.status_code == 200
        assert "key2" in b64_unpickle(response.json().get("data"))

    def test_delete_obj(self, http_client):
        # https://www.python-httpx.org/compatibility/#request-body-on-http-methods
        keys = ["key2"]
        data = pickle_b64(keys)
        response = http_client.request("delete", url="/object", json={"data": data})
        assert response.status_code == 200

    def test_add_secrets(self, http_client):
        secrets = {"aws": "abc123"}
        data = pickle_b64(secrets)
        response = http_client.post("/secrets", json={"data": data})
        assert response.status_code == 200

    def test_call_module_method(self, http_client, base_cluster):
        # Create new func on the cluster, then call it
        remote_func = rh.function(summer).to(base_cluster)

        method_name = "call"
        module_name = remote_func.name
        args = (1, 2)
        kwargs = {"force": False}

        response = http_client.post(
            f"{module_name}/{method_name}",
            json={
                "data": pickle_b64([args, kwargs]),
                "env": None,
                "stream_logs": True,
                "save": False,
                "key": None,
                "remote": False,
                "run_async": False,
            },
        )
        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_async_call(self, async_http_client, base_cluster):
        remote_func = rh.function(summer).to(base_cluster)
        method = "call"

        response = await async_http_client.post(
            f"/call/{remote_func.name}/{method}?serialization=None",
            json={"args": [1, 2]},
        )
        assert response.status_code == 200


class TestTLSCertConfig(unittest.TestCase):
    def setUp(self):
        self.cert_config = TLSCertConfig()

    def test_generate_certs(self):
        # Generate certificates for a given address
        address = "127.0.0.1"
        self.cert_config.generate_certs(address=address)

        self.assertTrue(
            Path(self.cert_config.cert_path).exists(),
            "Certificate file was not created.",
        )

        self.assertTrue(
            Path(self.cert_config.key_path).exists(),
            "Private key file was not created.",
        )

        # Load the certificate and check properties
        with open(self.cert_config.cert_path, "rb") as cert_file:
            cert_data = cert_file.read()
            certificate = load_pem_x509_certificate(cert_data, default_backend())
            self.assertEqual(
                certificate.subject.get_attributes_for_oid(NameOID.COMMON_NAME)[
                    0
                ].value,
                "run.house",
            )
            self.assertIn(
                ipaddress.IPv4Address(address),
                certificate.extensions.get_extension_for_class(
                    x509.SubjectAlternativeName
                ).value.get_values_for_type(x509.IPAddress),
            )

        # Load the private key and check type
        with open(self.cert_config.key_path, "rb") as key_file:
            key_data = key_file.read()
            private_key = load_pem_private_key(
                key_data, password=None, backend=default_backend()
            )
            self.assertTrue(
                isinstance(private_key, rsa.RSAPrivateKey),
                "Private key is not an RSA key.",
            )

    @patch("os.path.abspath")
    @patch("os.path.expanduser")
    def test_resolve_absolute_path(self, mock_expanduser, mock_abspath):
        # Mock the expanduser and abspath to return a mock path
        mock_expanduser.return_value = "/mocked/home/user/ssl/certs/rh_server.crt"
        mock_abspath.return_value = "/mocked/absolute/path/to/ssl/certs/rh_server.crt"

        # Call the resolve_absolute_path function
        resolved_path = resolve_absolute_path(self.cert_config.cert_path)

        # Check that the mocked functions were called with the expected arguments
        mock_expanduser.assert_called_once_with(self.cert_config.cert_path)
        mock_abspath.assert_called_once_with(mock_expanduser.return_value)

        # Check that the resolved path matches the mock abspath return value
        self.assertEqual(resolved_path, mock_abspath.return_value)

    def tearDown(self):
        # Clean up the generated files
        Path(self.cert_config.cert_path).unlink(missing_ok=True)
        Path(self.cert_config.key_path).unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
