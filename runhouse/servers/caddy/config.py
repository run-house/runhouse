import logging
import subprocess
import textwrap
from pathlib import Path

logger = logging.getLogger(__name__)

from runhouse.constants import DEFAULT_SERVER_PORT


class CaddyConfig:
    BASE_CONFIG_PATH = "/etc/caddy/Caddyfile"
    RH_SERVER_PORT = 32300

    # Helpful commands:
    # sudo apt-get install net-tools
    # sudo netstat -tulpn | grep 443

    # Useful Caddy service commands:
    # sudo systemctl start caddy
    # sudo systemctl stop caddy
    # sudo systemctl status caddy
    # sudo systemctl reload caddy

    # For viewing logs (if running as a service):
    # journalctl -u caddy --no-pager | less +G

    # Useful Caddy background process commands:
    # caddy start
    # caddy stop
    # caddy validate --config /etc/caddy/Caddyfile

    # Checking config settings:
    # caddy adapt --config /etc/caddy/Caddyfile

    def __init__(
        self,
        address: str,
        rh_server_port: int = None,
        ssl_cert_path: str = None,
        ssl_key_path: str = None,
        use_https=False,
        force_reinstall=False,
    ):
        self.use_https = use_https
        self.rh_server_port = rh_server_port or DEFAULT_SERVER_PORT
        self.force_reinstall = force_reinstall

        # To expose the server to the internet, set address to the public IP, otherwise leave it as localhost
        self.address = address or "localhost"

        self.ssl_cert_path = Path(ssl_cert_path).expanduser() if ssl_cert_path else None
        self.ssl_key_path = Path(ssl_key_path).expanduser() if ssl_key_path else None

        if self.use_https:
            # If using https, need to provide certs
            if self.ssl_cert_path is None:
                raise ValueError(
                    "No SSL cert path provided. Cannot enable HTTPS without a domain or custom certs."
                )
            if not self.ssl_cert_path.exists():
                raise FileNotFoundError(
                    f"Failed to find SSL cert file in path: {self.ssl_cert_path}"
                )

            if self.ssl_key_path is None:
                raise ValueError(
                    "No SSL key path provided. Cannot enable HTTPS without a domain or custom certs."
                )

            if not self.ssl_key_path.exists():
                raise FileNotFoundError(
                    f"Failed to find SSL key file in path: {self.ssl_key_path}"
                )

    @property
    def caddyfile(self):
        """Caddy config file."""
        return Path(self.BASE_CONFIG_PATH).expanduser()

    def configure(self):
        """Configure Caddy to proxy requests to the Fast API HTTP server"""
        if not self._is_configured() or self.force_reinstall:
            logger.info(f"Configuring Caddy for address: {self.address}")
            self._install()
            self._build_template()
            self._start_caddy()

        # Reload Caddy with the updated config
        logger.info("Reloading Caddy")
        self.reload()

        if not self._is_configured():
            raise RuntimeError("Failed to configure and start Caddy")

        logger.info("Successfully configured Caddy")

    # -----------------
    # HELPERS
    # -----------------
    def reload(self):
        # https://caddyserver.com/docs/command-line#caddy-stop
        reload_cmd = ["sudo", "systemctl", "reload", "caddy"]
        result = subprocess.run(
            reload_cmd,
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            if "systemctl: command not found" in result.stderr:
                # If running in a docker container or distro without systemctl, reload caddy as a background process
                reload_cmd = f"caddy reload --config {str(self.caddyfile)}"
                try:
                    subprocess.run(reload_cmd, shell=True, check=True, text=True)
                except subprocess.CalledProcessError as e:
                    raise RuntimeError(
                        f"Failed to reload Caddy as a background process: {e}"
                    )
            else:
                raise RuntimeError(f"Failed to reload Caddy service: {result.stderr}")

        logger.info("Successfully reloaded Caddy.")

    def _install(self):
        check_cmd = ["sudo", "caddy", "version"]
        result = subprocess.run(
            check_cmd,
            capture_output=True,
            text=True,
        )
        if result.returncode == 0 and "v2." in result.stdout:
            logger.info("Caddy is already installed, skipping install.")
        else:
            # Install caddy as a service (or background process if we can't use systemctl)
            # https://caddyserver.com/docs/running#using-the-service
            logger.info("Installing Caddy.")

            commands = [
                "sudo apt update",
                "sudo apt install -y debian-keyring debian-archive-keyring apt-transport-https",
                "yes | curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/gpg.key' | sudo gpg --dearmor -o /usr/share/keyrings/caddy-stable-archive-keyring.gpg",  # noqa
                "yes | curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/debian.deb.txt' | sudo tee /etc/apt/sources.list.d/caddy-stable.list",  # noqa
                "sudo apt update",
                "sudo apt install caddy",
            ]

            for cmd in commands:
                try:
                    subprocess.run(cmd, shell=True, check=True, text=True)
                except subprocess.CalledProcessError as e:
                    raise RuntimeError(f"Failed to run Caddy install command: {e}")

        # "certutil" is required for generating certs
        if not self.ssl_key_path and not self.ssl_cert_path:
            cert_lib_cmd = ["sudo", "apt", "install", "libnss3-tools"]
            result = subprocess.run(
                cert_lib_cmd,
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                logger.warning(
                    f"Could not installed certutil, skipping: {result.stderr}"
                )

        logger.info("Successfully installed Caddy.")

    def _http_template(self):
        return textwrap.dedent(
            f"""
            {{
                default_sni {self.address}
            }}

            http://{self.address} {{
                reverse_proxy localhost:{self.rh_server_port}
            }}
        """
        ).strip()

    def _https_template(self):
        if self.ssl_key_path and self.ssl_cert_path:
            logger.info("Using custom certs to enable HTTPs")
            tls_directive = f"tls {self.ssl_cert_path} {self.ssl_key_path}"
            address_or_domain = self.address
        else:
            # Do not support issuing self-signed certs on the cluster
            # Unverified certs should be generated client side and passed in as custom certs
            raise RuntimeError("No certs provided. Cannot enable HTTPS.")

        return textwrap.dedent(
            f"""
            {{
                default_sni {address_or_domain}
            }}

            https://{address_or_domain} {{
                {tls_directive}
                reverse_proxy 127.0.0.1:{self.rh_server_port}
            }}
            """
        ).strip()

    def _build_template(self):
        if self.use_https:
            # Update firewall rule for HTTPS
            subprocess.run(
                "sudo ufw allow 443/tcp",
                check=True,
                capture_output=True,
                text=True,
                shell=True,
            )
        else:
            # Update firewall rule for HTTP
            subprocess.run(
                "sudo ufw allow 80/tcp",
                check=True,
                capture_output=True,
                text=True,
                shell=True,
            )

        try:
            template = (
                self._https_template() if self.use_https else self._http_template()
            )

            # Update the Caddyfile with this new template
            subprocess.run(
                f"echo '{template}' | sudo tee {self.caddyfile} > /dev/null",
                shell=True,
                check=True,
                capture_output=True,
                text=True,
            )

        except Exception as e:
            raise e

        logger.info("Successfully built and formatted Caddy template.")

    def _is_configured(self) -> bool:
        logger.info("Checking Caddy configuration.")
        result = subprocess.run(
            ["sudo", "systemctl", "status", "caddy"],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            if "systemctl: command not found" in result.stderr:
                # If running in a docker container or distro without systemctl, check whether Caddy has been configured
                run_cmd = f"caddy validate --config {str(self.caddyfile)}"
                try:
                    subprocess.run(run_cmd, shell=True, check=True, text=True)
                except subprocess.CalledProcessError as e:
                    logger.warning(e)
                    return False
                return True

            else:
                logger.warning(result.stderr)
                return False

        return "active (running)" in result.stdout

    def _start_caddy(self):
        """Run the caddy server as a service or background process. Try to start as a service first, if that
        fails then default to running as a background process."""
        # Add Caddy as a sudoer, otherwise will not be able to install certs on the server
        # Will receive an error that looks like:
        # caddy : user NOT in sudoers ; TTY=unknown ; PWD=/ ; USER=root
        # https://github.com/caddyserver/caddy/issues/4248
        logger.info("Adding Caddy as trusted app.")
        try:
            subprocess.run(
                "sudo mkdir -p /var/lib/caddy/.local && "
                "sudo chown -R caddy: /var/lib/caddy",
                shell=True,
                check=True,
                text=True,
            )
        except subprocess.CalledProcessError as e:
            raise e

        logger.info("Starting Caddy.")
        run_cmd = ["sudo", "systemctl", "start", "caddy"]
        result = subprocess.run(
            run_cmd,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            if "systemctl: command not found" in result.stderr:
                # If running in a docker container or distro without systemctl, we need to start Caddy manually
                # as a background process
                run_cmd = "caddy start"
                try:
                    subprocess.run(run_cmd, shell=True, check=True, text=True)
                except subprocess.CalledProcessError as e:
                    raise RuntimeError(
                        f"Failed to start Caddy as a background process: {e}"
                    )
            else:
                raise RuntimeError(f"Failed to run Caddy service: {result.stderr}")

        logger.info("Successfully started Caddy.")
