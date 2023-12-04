import datetime

import ipaddress
import os
import ssl
import threading
import unittest

from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from unittest.mock import patch

import pytest
import requests
from cryptography import x509
from cryptography.hazmat.backends import default_backend

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives.serialization import load_pem_private_key
from cryptography.x509 import load_pem_x509_certificate
from cryptography.x509.oid import NameOID
from requests.exceptions import SSLError

from runhouse.rns.utils.api import resolve_absolute_path
from runhouse.servers.http.certs import TLSCertConfig


class SimpleHTTPRequestHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"Hello, SSL!")


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


class TestTLSCertConfig:
    @pytest.fixture(autouse=True)
    def init_fixtures(self):
        self.cert_config = TLSCertConfig()

        yield

        # Clean up the generated files
        Path(self.cert_config.cert_path).unlink(missing_ok=True)
        Path(self.cert_config.key_path).unlink(missing_ok=True)

    @pytest.mark.level("unit")
    def test_generate_certs(self):
        # Generate certificates for a given address
        address = "127.0.0.1"
        self.cert_config.generate_certs(address=address)

        assert Path(self.cert_config.cert_path).exists()
        assert Path(self.cert_config.key_path).exists()

        # Load the certificate and check properties
        with open(self.cert_config.cert_path, "rb") as cert_file:
            cert_data = cert_file.read()
            certificate = load_pem_x509_certificate(cert_data, default_backend())
            assert (
                certificate.subject.get_attributes_for_oid(NameOID.COMMON_NAME)[0].value
                == "run.house"
            )

            assert ipaddress.IPv4Address(
                address
            ) in certificate.extensions.get_extension_for_class(
                x509.SubjectAlternativeName
            ).value.get_values_for_type(
                x509.IPAddress
            )

        # Load the private key and check type
        with open(self.cert_config.key_path, "rb") as key_file:
            key_data = key_file.read()
            private_key = load_pem_private_key(
                key_data, password=None, backend=default_backend()
            )
            assert isinstance(
                private_key, rsa.RSAPrivateKey
            ), "Private key is not an RSA key."

    @pytest.mark.level("unit")
    @patch("os.path.abspath")
    @patch("os.path.expanduser")
    def test_resolve_absolute_path(self, mock_expanduser, mock_abspath):
        # Mock the expanduser and abspath to return a mock path
        mock_expanduser.return_value = "/mocked/home/user/ssl/certs/rh_server.crt"
        mock_abspath.return_value = "/mocked/absolute/path/to/ssl/certs/rh_server.crt"

        resolved_path = resolve_absolute_path(self.cert_config.cert_path)

        mock_expanduser.assert_called_once_with(self.cert_config.cert_path)
        mock_abspath.assert_called_once_with(mock_expanduser.return_value)

        assert resolved_path == mock_abspath.return_value


class TestHTTPSCertValidity:
    @pytest.fixture(autouse=True)
    def init_fixtures(self):
        self.cert_file = "test_cert.pem"
        self.key_file = "test_key.pem"
        self._generate_test_certificate(self.cert_file, self.key_file)

        # Start a local HTTPS server using the self-signed certificate
        self.server, self.port, self.server_thread = create_test_https_server(
            self.cert_file, self.key_file
        )

        yield

        # Clean up the server and the generated certificate files
        self.server.shutdown()
        self.server.server_close()
        self.server_thread.join()
        os.remove(self.cert_file)
        os.remove(self.key_file)

    def _generate_test_certificate(self, cert_file, key_file):
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

    def verify_cert(self, cert_path):
        if not cert_path:
            return False

        cert_path = Path(cert_path)
        if not cert_path.exists():
            return False

        # Check whether the cert is self-signed, if so we cannot use verification
        with open(cert_path, "rb") as cert_file:
            cert = x509.load_pem_x509_certificate(cert_file.read(), default_backend())

        if cert.issuer == cert.subject:
            return False

        return True

    @pytest.mark.level("unit")
    def test_https_request_with_cert_verification(self):
        response = requests.get(f"https://localhost:{self.port}", verify=self.cert_file)

        assert response.status_code == 200
        assert response.text == "Hello, SSL!"

    @pytest.mark.level("unit")
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

    def test_https_request_with_self_signed_cert(self):
        should_verify = self.verify_cert(self.cert_file)

        # If the certificate is self-signed, verification should be False
        assert not should_verify

        response = requests.get(
            f"https://localhost:{self.port}",
            verify=should_verify,
        )

        assert response.status_code == 200
        assert response.text == "Hello, SSL!"

    @pytest.mark.level("unit")
    @patch("requests.get")
    def test_https_request_with_verified_cert(self, mock_get):
        # Mock the response of the requests.get call
        mock_response = requests.Response()
        mock_response.status_code = 200
        mock_response._content = b"Hello, SSL!"
        mock_get.return_value = mock_response

        # Simulate the scenario where the certificate should be verified
        should_verify = True

        # Call the method under test
        response = requests.get(f"https://localhost:{self.port}", verify=should_verify)

        mock_get.assert_called_once_with(
            f"https://localhost:{self.port}", verify=should_verify
        )

        assert response.status_code == 200
        assert response.text == "Hello, SSL!"


if __name__ == "__main__":
    unittest.main()
