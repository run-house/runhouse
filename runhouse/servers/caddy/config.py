import logging
import subprocess
import textwrap
from pathlib import Path

logger = logging.getLogger(__name__)

from runhouse.constants import DEFAULT_SERVER_PORT


class CaddyConfig:
    BASE_CONFIG_PATH = "~/.rh/caddy/Caddyfile"
    RH_SERVER_PORT = 32300

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

        self.ssl_cert_path = ssl_cert_path
        self.ssl_key_path = ssl_key_path

        self.force_reinstall = force_reinstall

        # To expose the server to the internet, set address to the public IP, otherwise leave it as localhost
        self.address = address or "localhost"

    @property
    def caddyfile(self):
        return Path(self.BASE_CONFIG_PATH).expanduser()

    @property
    def exec_path(self):
        # TODO: probably easier to update the python path
        # e.g.: /home/ubuntu/.local/bin/caddy
        return Path("~/.local/bin/caddy").expanduser()

    def configure(self):
        """Configure Caddy to proxy requests to the Fast API HTTP server"""
        if not self.caddyfile.exists():
            logger.info(
                f"Creating Caddy config folder in path: {self.caddyfile.parent}"
            )
            self.caddyfile.parent.mkdir(parents=True, exist_ok=True)

            self._install()
            self._build_template()
            self._start_caddy()

        is_configured = self._is_configured()

        if not self.force_reinstall and is_configured:
            logger.info("Caddy is already configured")
            return

        # Reload Caddy with the updated config
        logger.info("Reloading Caddy config")
        self._reload()

        if not self._is_configured():
            raise RuntimeError("Failed to configure Caddy")

        logger.info("Successfully configured Caddy")

    # -----------------
    # HELPERS
    # -----------------

    def _reload(self):
        reload_cmd = [
            "sudo",
            str(self.exec_path),
            "reload",
            "--config",
            str(self.caddyfile),
        ]
        result = subprocess.run(
            reload_cmd,
            check=True,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Failed to reload Caddy: {result.stderr}")

    def _install(self):
        result = subprocess.run(
            "sudo curl -sS https://webi.sh/caddy | sh",
            shell=True,
            check=True,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Failed to install Caddy: {result.stderr}")

        logger.info("Successfully installed Caddy.")

    def _http_template(self):
        return textwrap.dedent(
            f"""
            {{
                admin :2019
            }}

            http://{self.address} {{
                reverse_proxy localhost:{self.rh_server_port}
            }}
        """
        ).strip()

    def _https_template(self):
        # Note: We cannot use fully managed certs since we only have an IP address, not a domain name
        # https://caddyserver.com/docs/automatic-https#hostname-requirements
        if self.ssl_key_path and self.ssl_cert_path:
            logger.info("Using custom certs for HTTPS")
            tls_directive = f"tls {self.ssl_cert_path} {self.ssl_key_path}"
        else:
            # Generate self-signed certs which will not be trusted by external clietns
            logger.info("Generating self-signed certs for HTTPS")
            tls_directive = "tls internal"

        return textwrap.dedent(
            f"""
            {{
                admin :2019
            }}

            https://{self.address} {{
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

        try:
            template = (
                self._https_template() if self.use_https else self._http_template()
            )
            logger.info(f"Template for Caddyfile\n {template}")
            with open(self.caddyfile, "w") as f:
                f.write(template)

        except Exception as e:
            raise RuntimeError(
                f"Error configuring new Caddy template (https={self.use_https}): {e}"
            )

        logger.info("Successfully built Caddy template.")

    def _is_configured(self) -> bool:
        logger.info("Checking Caddy configuration.")
        result = subprocess.run(
            ["sudo", str(self.exec_path), "validate", "--config", str(self.caddyfile)],
            check=True,
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            logger.error(result.stderr)
            return False

        return "Valid configuration" in result.stdout

    def _start_caddy(self):
        logger.info("Starting Caddy server.")
        # Run the caddy server as a background process
        run_cmd = [
            "sudo",
            str(self.exec_path),
            "start",
            "--config",
            str(self.caddyfile),
        ]
        result = subprocess.run(
            run_cmd,
            check=True,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Failed to run Caddy config: {result.stderr}")

        logger.info("Successfully applied Caddy config settings.")
