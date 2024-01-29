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

    # For viewing logs:
    # journalctl -u caddy --no-pager | less +G

    # Caddy service commands:
    # sudo systemctl start caddy
    # sudo systemctl stop caddy
    # sudo systemctl status caddy
    # sudo systemctl reload caddy

    # Checking config settings:
    # caddy adapt --config /etc/caddy/Caddyfile

    def __init__(
        self,
        address: str,
        domain: str,
        rh_server_port: int = None,
        ssl_cert_path: str = None,
        ssl_key_path: str = None,
        use_https=False,
        force_reinstall=False,
    ):
        self.use_https = use_https
        self.rh_server_port = rh_server_port or DEFAULT_SERVER_PORT

        self.ssl_cert_path = (
            Path(ssl_cert_path).expanduser()
            if (ssl_cert_path and self.use_https)
            else None
        )
        self.ssl_key_path = (
            Path(ssl_key_path).expanduser()
            if (ssl_key_path and self.use_https)
            else None
        )

        if self.ssl_cert_path and not self.ssl_cert_path.exists() and not domain:
            raise FileNotFoundError(
                f"Failed to find SSL cert file in path: {self.ssl_cert_path}"
            )

        if self.ssl_key_path and not self.ssl_key_path.exists() and not domain:
            raise FileNotFoundError(
                f"Failed to find SSL cert file in path: {self.ssl_key_path}"
            )

        self.force_reinstall = force_reinstall
        self.domain = domain

        # To expose the server to the internet, set address to the public IP, otherwise leave it as localhost
        self.address = address or "localhost"

    @property
    def caddyfile(self):
        return Path(self.BASE_CONFIG_PATH).expanduser()

    def configure(self):
        """Configure Caddy to proxy requests to the Fast API HTTP server"""
        if not self._is_configured() or self.force_reinstall:
            logger.info(f"Configuring Caddy (force reinstall={self.force_reinstall})")
            self._install()
            self._build_template()
            self._start_caddy()

        # Reload Caddy with the updated config
        logger.info("Reloading Caddy service")
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
            raise RuntimeError(f"Failed to reload Caddy: {result.stderr}")

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
            # Install caddy as a service
            # https://caddyserver.com/docs/running#using-the-service
            logger.info("Installing Caddy as a service.")

            commands = [
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
                    raise RuntimeError(f"Failed to install Caddy as a service: {e}")

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

        logger.info("Successfully installed Caddy as service.")

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
            logger.info("Using custom certs for HTTPS")
            tls_directive = f"tls {self.ssl_cert_path} {self.ssl_key_path}"
            address_or_domain = self.address
        elif self.domain:
            # https://caddyserver.com/docs/automatic-https#hostname-requirements
            logger.info(f"Using Caddy to generate certs for domain: {self.domain}")
            tls_directive = "tls on_demand"
            address_or_domain = self.domain
        else:
            logger.warning(
                "No domain or custom certs specified, issuing self-signed certs"
            )
            tls_directive = "tls internal"
            address_or_domain = self.address

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
        # Update firewall rule where relevant
        subprocess.run(
            "sudo ufw allow 443/tcp",
            check=True,
            capture_output=True,
            text=True,
            shell=True,
        )
        logger.info("Updated ufw firewall rule to allow HTTPS traffic")

        if self.ssl_cert_path and self.ssl_key_path:
            subprocess.run(
                f"sudo chmod 600 {self.ssl_cert_path} && "
                f"sudo chmod 600 {self.ssl_key_path}",
                shell=True,
                check=True,
                capture_output=True,
                text=True,
            )

        else:
            # Add Caddy as a sudoer, otherwise will not be able to install certs on the server
            # Will receive an error that looks like:
            # caddy : user NOT in sudoers ; TTY=unknown ; PWD=/ ; USER=root
            logger.info("Adding Caddy to the list of trusted applications.")
            result = subprocess.run(
                ["sudo", "caddy", "trust"],
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                raise RuntimeError(
                    f"Failed to add Caddy to trusted apps: {result.stderr}"
                )

        try:
            template = (
                self._https_template() if self.use_https else self._http_template()
            )
            logger.info(f"New template for Caddyfile:\n {template}")

            # Update the base (default) Caddyfile with this new template
            subprocess.run(
                f"echo '{template}' | sudo tee {self.caddyfile} > /dev/null",
                shell=True,
                check=True,
                capture_output=True,
                text=True,
            )

        except Exception as e:
            raise e

        # format the Caddyfile to remove spammy warnings
        result = subprocess.run(
            ["sudo", "caddy", "fmt", "--overwrite"],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            logger.warning(f"Failed to format Caddy template: {result.stderr}")

        logger.info(
            f"Successfully built and formatted Caddy template (https={self.use_https})."
        )

    def _is_configured(self) -> bool:
        logger.info("Checking Caddy configuration.")
        result = subprocess.run(
            ["sudo", "systemctl", "status", "caddy"],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            logger.warning(result.stderr)
            return False

        return "active (running)" in result.stdout

    def _start_caddy(self):
        # Run the caddy server as a background service
        logger.info("Starting Caddy service.")
        run_cmd = ["sudo", "systemctl", "start", "caddy"]
        result = subprocess.run(
            run_cmd,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Failed to run Caddy service: {result.stderr}")

        logger.info("Successfully applied Caddy service settings.")
