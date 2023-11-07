import datetime
import os
import ssl
import threading
import unittest
import warnings

from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest
import requests
from cryptography import x509
from cryptography.hazmat.backends import default_backend

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID
from requests.exceptions import SSLError


# TODO [JL] use docker container fixture to test this?
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

    def _use_cert_verification(self, cert_path):
        if not cert_path:
            return False

        cert_path = Path(cert_path)
        if not cert_path.exists():
            return False

        # Check whether the cert is self-signed, if so we cannot use verification
        with open(cert_path, "rb") as cert_file:
            cert = x509.load_pem_x509_certificate(cert_file.read(), default_backend())

        if cert.issuer == cert.subject:
            warnings.warn(
                f"Cert in use ({cert_path}) is self-signed, cannot verify in requests to server."
            )
            return False

        return True

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

    def test_https_request_with_self_signed_cert(self):
        # Determine if we should verify the requests to the server
        should_verify = self._use_cert_verification(self.cert_file)

        # If the certificate is self-signed, verification should be False
        self.assertFalse(should_verify)

        # Make a request to the server without verification if the cert is self-signed
        response = requests.get(
            f"https://localhost:{self.port}",
            verify=False if not should_verify else self.cert_file,
        )

        # If no exception is raised, the request is successful
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.text, "Hello, SSL!")


if __name__ == "__main__":
    unittest.main()
