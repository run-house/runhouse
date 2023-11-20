import datetime
import ipaddress
import logging
from pathlib import Path

from cryptography import x509
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID

from runhouse.rns.utils.api import resolve_absolute_path

logger = logging.getLogger(__name__)


class TLSCertConfig:
    """Handler for creating and managing the TLS certs needed to enable HTTPS on the Runhouse API server.
    Note that this class can be initialized both locally and on the cluster, which affects the default file
    paths for storing the generated certs."""

    CERT_NAME = "rh_server.crt"
    PRIVATE_KEY_NAME = "rh_server.key"
    TOKEN_VALIDITY_DAYS = 365

    # Base directory for certs on both the cluster and locally
    DEFAULT_PRIVATE_KEY_DIR = "~/ssl/private"
    DEFAULT_CERT_DIR = "~/ssl/certs"

    def __init__(
        self, cert_path: str = None, key_path: str = None, dir_name: str = None
    ):
        self._cert_path = cert_path
        self._key_path = key_path

        # Useful for initializing locally, where the user may have multiple certs stored for different clusters
        # Each cluster will have its own directory for storing the cert / private key files
        self.dir_name = dir_name

    @property
    def cert_path(self):
        if self._cert_path is not None:
            return resolve_absolute_path(self._cert_path)

        if not self.dir_name:
            # Default cert path when initializing on a cluster
            return str(Path(f"{self.DEFAULT_CERT_DIR}/{self.CERT_NAME}").expanduser())
        else:
            # Default cert path when initializing locally - certs to be saved locally in a folder dedicated to the
            # relevant cluster
            return str(
                Path(
                    f"{self.DEFAULT_CERT_DIR}/{self.dir_name}/{self.CERT_NAME}"
                ).expanduser()
            )

    @cert_path.setter
    def cert_path(self, cert_path):
        self._cert_path = cert_path

    @property
    def key_path(self):
        if self._key_path is not None:
            return resolve_absolute_path(self._key_path)

        if not self.dir_name:
            # Default cert path when initializing on a cluster
            return str(
                Path(
                    f"{self.DEFAULT_PRIVATE_KEY_DIR}/{self.PRIVATE_KEY_NAME}"
                ).expanduser()
            )
        else:
            # Default cert path when initializing locally
            return str(
                Path(
                    f"{self.DEFAULT_PRIVATE_KEY_DIR}/{self.dir_name}/{self.PRIVATE_KEY_NAME}"
                ).expanduser()
            )

    @key_path.setter
    def key_path(self, key_path):
        self._key_path = key_path

    @property
    def cert_dir(self):
        return Path(self.cert_path).parent

    @property
    def key_dir(self):
        return Path(self.key_path).parent

    def generate_certs(self, address: str = None):
        """Create a self-signed SSL certificate. This won't be verified by a CA, but the connection will
        still be encrypted."""
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

        if address is not None:
            try:
                # Check if the address is a valid IP address
                ip_addr = ipaddress.IPv4Address(address)
                subject_names.append(x509.IPAddress(ip_addr))
            except ipaddress.AddressValueError:
                # If not a valid IP address (e.g. "localhost"), treat it as a DNS name
                subject_names.append(x509.DNSName(address))

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
