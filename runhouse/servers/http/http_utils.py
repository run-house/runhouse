import codecs
import datetime
import hashlib
import ipaddress
import logging
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from cryptography import x509
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID

from pydantic import BaseModel
from ray import cloudpickle as pickle

from runhouse.rns.utils.api import resolve_absolute_path

logger = logging.getLogger(__name__)

DEFAULT_SERVER_HOST = "127.0.0.1"
DEFAULT_SERVER_PORT = 32300


class Message(BaseModel):
    data: str = None
    env: str = None
    key: Optional[str] = None
    stream_logs: Optional[bool] = True
    save: Optional[bool] = False
    remote: Optional[bool] = False
    run_async: Optional[bool] = False


class Args(BaseModel):
    args: Optional[List[Any]]
    kwargs: Optional[Dict[str, Any]]


class Response(BaseModel):
    data: Union[None, str, List[str], Dict]
    error: Optional[str]
    traceback: Optional[str]
    output_type: str


class OutputType:
    EXCEPTION = "exception"
    STDOUT = "stdout"
    STDERR = "stderr"
    SUCCESS = "success"  # No output
    NOT_FOUND = "not_found"
    CANCELLED = "cancelled"
    RESULT = "result"
    RESULT_LIST = "result_list"
    RESULT_STREAM = "result_stream"
    SUCCESS_STREAM = "success_stream"  # No output, but with generators
    CONFIG = "config"


class TLSCertConfig:
    CERT_NAME = "rh_server.crt"
    PRIVATE_KEY_NAME = "rh_server.key"
    TOKEN_VALIDITY_DAYS = 365
    DEFAULT_PRIVATE_KEY_DIR = "~/ssl/private"
    DEFAULT_CERT_DIR = "~/ssl/certs"

    def __init__(
        self,
        cert_path: str = None,
        key_path: str = None,
        cluster_name: str = None,
    ):

        self._cert_path = cert_path
        self._key_path = key_path

        # Need to indicate whether we are on the cluster or not (we can't use rh.here yet
        # bc this is part of the HTTP server init)
        self.cluster_name = cluster_name

    @property
    def cert_path(self):
        if self._cert_path is not None:
            return resolve_absolute_path(self._cert_path)

        if not self.cluster_name:
            # Default cert path when initializing on a cluster
            return str(Path(f"{self.DEFAULT_CERT_DIR}/{self.CERT_NAME}").expanduser())
        else:
            # Default cert path when initializing locally - certs to be saved locally in a folder dedicated to the
            # relevant cluster
            return str(
                Path(
                    f"{self.DEFAULT_CERT_DIR}/{self.cluster_name}/{self.CERT_NAME}"
                ).expanduser()
            )

    @cert_path.setter
    def cert_path(self, cert_path):
        self._cert_path = cert_path

    @property
    def key_path(self):
        if self._key_path is not None:
            return resolve_absolute_path(self._key_path)

        if not self.cluster_name:
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
                    f"{self.DEFAULT_PRIVATE_KEY_DIR}/{self.cluster_name}/{self.PRIVATE_KEY_NAME}"
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
            subject_names.append(x509.IPAddress(ipaddress.IPv4Address(address)))

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
        """Save the private key and cert files to the cluster's file system."""
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


class ServerCache:
    AUTH_CACHE = {}

    @classmethod
    def get_resources(cls, token: str) -> dict:
        """Get resources associated with a particular user's token"""
        return cls.AUTH_CACHE.get(cls.hash_token(token), {})

    @classmethod
    def put_resources(cls, token: str, resources: dict):
        """Update server cache with a user's resources and access type"""
        cls.AUTH_CACHE[cls.hash_token(token)] = resources

    @staticmethod
    def hash_token(token: str) -> str:
        """Hash the user's token to avoid storing them in plain text on the cluster."""
        return hashlib.sha256(token.encode()).hexdigest()


def pickle_b64(picklable):
    return codecs.encode(pickle.dumps(picklable), "base64").decode()


def b64_unpickle(b64_pickled):
    return pickle.loads(codecs.decode(b64_pickled.encode(), "base64"))


def handle_response(response_data, output_type, err_str):
    if output_type in [OutputType.RESULT, OutputType.RESULT_STREAM]:
        return b64_unpickle(response_data["data"])
    elif output_type == OutputType.CONFIG:
        # No need to unpickle since this was just sent as json
        return response_data["data"]
    elif output_type == OutputType.RESULT_LIST:
        # Map, starmap, and repeat return lists of results
        return [b64_unpickle(val) for val in response_data["data"]]
    elif output_type == OutputType.NOT_FOUND:
        raise KeyError(f"{err_str}: key {response_data['data']} not found")
    elif output_type == OutputType.CANCELLED:
        raise RuntimeError(f"{err_str}: task was cancelled")
    elif output_type in [OutputType.SUCCESS, OutputType.SUCCESS_STREAM]:
        return
    elif output_type == OutputType.EXCEPTION:
        fn_exception = b64_unpickle(response_data["error"])
        fn_traceback = b64_unpickle(response_data["traceback"])
        logger.error(f"{err_str}: {fn_exception}")
        logger.error(f"Traceback: {fn_traceback}")
        raise fn_exception
    elif output_type == OutputType.STDOUT:
        res = response_data["data"]
        # Regex to match tqdm progress bars
        tqdm_regex = re.compile(r"(.+)%\|(.+)\|\s+(.+)/(.+)")
        for line in res:
            if tqdm_regex.match(line):
                # tqdm lines are always preceded by a \n, so we can use \x1b[1A to move the cursor up one line
                # For some reason, doesn't work in PyCharm's console, but works in the terminal
                print("\x1b[1A\r" + line, end="", flush=True)
            else:
                print(line, end="", flush=True)
    elif output_type == OutputType.STDERR:
        res = response_data["data"]
        print(res, file=sys.stderr)
