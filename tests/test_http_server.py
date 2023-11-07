import datetime
import ipaddress
import os
import ssl
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer

from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

import httpx
import pytest
import requests

from cryptography import x509
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives.serialization import load_pem_private_key
from cryptography.x509 import load_pem_x509_certificate
from cryptography.x509.oid import NameOID

from fastapi.testclient import TestClient

from httpx import AsyncClient
from requests.exceptions import SSLError
from runhouse.rns.utils.api import resolve_absolute_path
from runhouse.servers.http.certs import TLSCertConfig
from runhouse.servers.http.http_server import app
from runhouse.servers.http.http_utils import b64_unpickle, pickle_b64
from runhouse.servers.nginx.config import NginxConfig

# Note: Server is assumed to be running on local docker container
BASE_URL = "http://localhost:32300"


@pytest.fixture(scope="module")
def test_client():
    # TODO [JL] see if we can get this to work with FastAPI's TestClient (instead of directly using httpx)?
    # (Might not work for making requests to a docker container)
    # https://fastapi.tiangolo.com/tutorial/testing/
    with TestClient(app, base_url=BASE_URL) as client:
        yield client


@pytest.fixture(scope="module")
def http_client():
    with httpx.Client(base_url=BASE_URL) as client:
        yield client


@unittest.skip("Not implemented yet.")
@pytest.fixture(scope="module")
async def async_client():
    async with AsyncClient(app=app, base_url=BASE_URL) as ac:
        yield ac


class TestHTTPServer:
    def test_get_cert(self, http_client):
        response = http_client.get("/cert")
        assert response.status_code == 200
        cert = b64_unpickle(response.json().get("data"))
        assert isinstance(cert, bytes)

    def test_check_server(self, http_client):
        response = http_client.get("/check")
        assert response.status_code == 200

    def test_put_resource(
        self, http_client, local_blob, local_docker_cluster_public_key
    ):
        state = None
        resource = local_blob.to(local_docker_cluster_public_key)
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
        keys = ["key2"]
        data = pickle_b64(keys)
        response = http_client.delete("/object", data=data)
        assert response.status_code == 200

    def test_add_secrets(self, http_client):
        secrets = {"aws": "abc123"}
        data = pickle_b64(secrets)
        response = http_client.post("/secrets", json={"data": data})
        assert response.status_code == 200

    def test_call_module_method(self, http_client):
        method_name = "install"  # "call"
        module_name = "base_env"  # "summer"
        args = ()  # (1, 2)
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

    @unittest.skip("Not implemented yet.")
    @pytest.mark.anyio
    async def test_call(self, async_client):
        response = await async_client.get("/call/{module}/{method}")
        assert response.status_code == 200
        assert response.json() == {"message": "Tomato"}


class TestNginxConfiguration(unittest.TestCase):
    def setUp(self):
        self.config = NginxConfig(
            address="127.0.0.1",
            rh_server_port=32300,
            http_port=80,
            https_port=443,
            ssl_key_path="/path/to/ssl.key",
            ssl_cert_path="/path/to/ssl.cert",
            force_reinstall=False,
        )

    @patch("builtins.open", new_callable=mock_open)
    @patch("pathlib.Path.exists")
    @patch("subprocess.run")
    def test_nginx_config_generation(
        self, mock_subprocess_run, mock_path_exists, mock_file_open
    ):
        # Assume the paths do not exist for this test
        mock_path_exists.return_value = False

        # Mock the subprocess.run to pretend the chmod command succeeds
        mock_subprocess_run.return_value = MagicMock(returncode=0)

        self.config._build_template()

        mock_file_open.assert_called_once_with(self.config.BASE_CONFIG_PATH, "w")

        file_handle = mock_file_open()

        # Assert that we wrote the template to the file
        template = (
            self.config._https_template()
            if self.config._use_https
            else self.config._http_template()
        )
        file_handle.write.assert_called_once_with(template)

        # Assert that subprocess.run was called to change permissions
        mock_subprocess_run.assert_called()

    @patch("subprocess.run")
    def test_nginx_reload(self, mock_subprocess_run):
        # Mock the subprocess.run to pretend the reload command succeeds
        mock_subprocess_run.return_value = MagicMock(returncode=0)

        self.config.reload()

        # Assert that the subprocess.run was called with the nginx reload command
        mock_subprocess_run.assert_called_once_with(
            "sudo systemctl reload nginx",
            shell=True,
            check=True,
            capture_output=True,
            text=True,
        )


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


class SimpleHTTPRequestHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"Hello, SSL!")


# TODO [JL] use separate docker container fixture configured for HTTPs to test this?
def create_test_https_server(
    cert_file, key_file, server_class=HTTPServer, handler_class=SimpleHTTPRequestHandler
):
    httpd = server_class(("localhost", 0), handler_class)
    httpd.socket = ssl.wrap_socket(
        httpd.socket, certfile=cert_file, keyfile=key_file, server_side=True
    )
    sa = httpd.socket.getsockname()
    thread = threading.Thread(target=httpd.serve_forever)
    thread.daemon = True
    thread.start()
    return httpd, sa[1], thread


class TestHTTPSCertValidity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Create a self-signed test certificate for the local server
        cls.cert_file = "test_cert.pem"
        cls.key_file = "test_key.pem"
        cls._generate_test_certificate(cls.cert_file, cls.key_file)

        # Start a local HTTPS server using the self-signed certificate
        cls.server, cls.port, cls.server_thread = create_test_https_server(
            cls.cert_file, cls.key_file
        )

    @classmethod
    def tearDownClass(cls):
        # Clean up the server and the generated certificate files
        cls.server.shutdown()
        cls.server.server_close()
        cls.server_thread.join()
        os.remove(cls.cert_file)
        os.remove(cls.key_file)

    @classmethod
    def _generate_test_certificate(cls, cert_file, key_file):
        private_key = rsa.generate_private_key(
            public_exponent=65537, key_size=2048, backend=default_backend()
        )

        with open(key_file, "wb") as f:
            f.write(
                private_key.private_bytes(
                    encoding=serialization.Encoding.PEM,
                    format=serialization.PrivateFormat.TraditionalOpenSSL,
                    encryption_algorithm=serialization.NoEncryption(),
                )
            )

        subject = issuer = x509.Name(
            [
                x509.NameAttribute(NameOID.COUNTRY_NAME, "US"),
                x509.NameAttribute(NameOID.STATE_OR_PROVINCE_NAME, "California"),
                x509.NameAttribute(NameOID.LOCALITY_NAME, "San Francisco"),
                x509.NameAttribute(NameOID.ORGANIZATION_NAME, "My Company"),
                x509.NameAttribute(NameOID.COMMON_NAME, "localhost"),
            ]
        )
        cert = (
            x509.CertificateBuilder()
            .subject_name(subject)
            .issuer_name(issuer)
            .public_key(private_key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(datetime.datetime.utcnow())
            .not_valid_after(datetime.datetime.utcnow() + datetime.timedelta(days=10))
            .add_extension(
                x509.SubjectAlternativeName([x509.DNSName("localhost")]),
                critical=False,
            )
            .sign(private_key, hashes.SHA256(), default_backend())
        )

        with open(cert_file, "wb") as f:
            f.write(cert.public_bytes(serialization.Encoding.PEM))

    def test_https_request_with_cert_verification(self):
        response = requests.get(f"https://localhost:{self.port}", verify=self.cert_file)

        # If no exception is raised, the cert is valid
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.text, "Hello, SSL!")

    def test_https_request_with_invalid_cert_verification(self):
        dummy_cert_path = "dummy_cert.pem"
        with open(dummy_cert_path, "w") as f:
            f.write("-----BEGIN CERTIFICATE-----\n")
            f.write("Invalid certificate content\n")
            f.write("-----END CERTIFICATE-----\n")

        # should raise an SSLError because the cert won't be valid for the server's SSL setup
        with pytest.raises(SSLError):
            requests.get(f"https://localhost:{self.port}", verify=dummy_cert_path)

        os.remove(dummy_cert_path)


if __name__ == "__main__":
    unittest.main()
