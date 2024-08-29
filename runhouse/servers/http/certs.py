import datetime
import ipaddress
from pathlib import Path

from cryptography import x509
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID

from runhouse.logger import get_logger

from runhouse.rns.utils.api import resolve_absolute_path

logger = get_logger(__name__)


class TLSCertConfig:
    """Handler for creating and managing the TLS certs needed to enable HTTPS on the Runhouse API server."""

    CERT_NAME = "rh_server.crt"
    PRIVATE_KEY_NAME = "rh_server.key"
    TOKEN_VALIDITY_DAYS = 365

    # Base directory for certs stored locally
    # Note: Each user will have their own set of certs, to be reused on for each cluster
    LOCAL_CERT_DIR = "~/.rh/certs"

    # https://caddy.community/t/permission-denied-error-when-caddy-try-to-save-the-certificate/15026
    # Note: When running as a systemd service, Caddy as runs as the "caddy" user and doesn’t have permission to
    # read files in the home directory. Easiest solution to ensure there are no permission errors is to save the
    # certs in the Caddy cluster directory when Caddy is enabled.
    CADDY_CLUSTER_DIR = "/var/lib/caddy"
    DEFAULT_CLUSTER_DIR = "~/certs"

    def __init__(self, cert_path: str = None, key_path: str = None):
        self._cert_path = cert_path
        self._key_path = key_path

    @property
    def cert_path(self):
        if self._cert_path is not None:
            return resolve_absolute_path(self._cert_path)

        return str(Path(f"{self.LOCAL_CERT_DIR}/{self.CERT_NAME}").expanduser())

    @cert_path.setter
    def cert_path(self, cert_path):
        self._cert_path = cert_path

    @property
    def key_path(self):
        if self._key_path is not None:
            return resolve_absolute_path(self._key_path)

        return str(Path(f"{self.LOCAL_CERT_DIR}/{self.PRIVATE_KEY_NAME}").expanduser())

    @key_path.setter
    def key_path(self, key_path):
        self._key_path = key_path

    @property
    def cert_dir(self):
        return Path(self.cert_path).parent

    @property
    def key_dir(self):
        return Path(self.key_path).parent

    def generate_certs(self, address: str = None, domain: str = None):
        """Create a self-signed SSL certificate. This won't be verified by a CA, but the connection will
        still be encrypted. If domain is provided, prioritize using it to allow the same cert to be re-used
        across multiple servers. If not, use the provided address (IP or DNS name)"""
        # Generate the private key
        private_key = rsa.generate_private_key(
            public_exponent=65537, key_size=2048, backend=default_backend()
        )

        # Create a self-signed certificate
        subject = issuer = x509.Name(
            [x509.NameAttribute(NameOID.COMMON_NAME, "run.house")]
        )

        subject_names = [
            x509.DNSName("localhost"),
            x509.IPAddress(ipaddress.IPv4Address("127.0.0.1")),
        ]

        if domain:
            # Add the provided domain name to the SAN
            subject_names.append(x509.DNSName(domain))

        elif address:
            try:
                # Check if the address is a valid IP address and add it to the SAN
                ip_addr = ipaddress.IPv4Address(address)
                subject_names.append(x509.IPAddress(ip_addr))
            except ipaddress.AddressValueError:
                # If not a valid IP address, treat it as an additional DNS name
                subject_names.append(x509.DNSName(address))
        else:
            raise ValueError(
                "At least one of `address` or `domain` must be provided to generate certs."
            )

        # Add Subject Alternative Name extension
        san = x509.SubjectAlternativeName(subject_names)

        cert = (
            x509.CertificateBuilder()
            .subject_name(subject)
            .issuer_name(issuer)
            .public_key(private_key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(datetime.datetime.utcnow())
            .not_valid_after(
                datetime.datetime.utcnow()
                + datetime.timedelta(days=self.TOKEN_VALIDITY_DAYS)
            )
            .add_extension(san, critical=False)
            .sign(private_key, hashes.SHA256(), default_backend())
        )

        self._write_cert_files(private_key, cert)

    def _write_cert_files(self, private_key, cert):
        """Save the private key and cert files on the system (either locally or on cluster)."""
        # Ensure the directories exist and have the correct permissions
        self.cert_dir.expanduser().mkdir(parents=True, mode=0o750, exist_ok=True)
        self.key_dir.expanduser().mkdir(parents=True, mode=0o750, exist_ok=True)

        with open(self.key_path, "wb") as key_file:
            key_file.write(
                private_key.private_bytes(
                    encoding=serialization.Encoding.PEM,
                    format=serialization.PrivateFormat.TraditionalOpenSSL,
                    encryption_algorithm=serialization.NoEncryption(),
                )
            )

        # Save the certificate to a file (server.crt)
        with open(self.cert_path, "wb") as cert_file:
            cert_file.write(cert.public_bytes(encoding=serialization.Encoding.PEM))

        logger.info(
            f"Certificate and private key files saved in paths: {self.cert_path} and {self.key_path}"
        )
