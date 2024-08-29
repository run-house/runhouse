import subprocess
import textwrap
from pathlib import Path

from runhouse.constants import DEFAULT_SERVER_PORT
from runhouse.logger import get_logger

logger = get_logger(__name__)
SYSTEMCTL_ERROR = "systemctl: command not found"


class CaddyConfig:
    BASE_CONFIG_PATH = "/etc/caddy/Caddyfile"
    RH_SERVER_PORT = 32300

    # Helpful commands:
    # sudo apt-get install net-tools
    # sudo netstat -tulpn | grep 443

    # For viewing logs:
    # journalctl -u caddy --no-pager | less +G

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
        domain: str = None,
        rh_server_port: int = None,
        ssl_cert_path: str = None,
        ssl_key_path: str = None,
        use_https=False,
        force_reinstall=False,
    ):
        self.use_https = use_https
        self.rh_server_port = rh_server_port or DEFAULT_SERVER_PORT
        self.domain = domain
        self.force_reinstall = force_reinstall

        # To expose the server to the internet, set address to the public IP, otherwise leave it as localhost
        self.address = address or "localhost"

        self.ssl_cert_path = Path(ssl_cert_path).expanduser() if ssl_cert_path else None
        self.ssl_key_path = Path(ssl_key_path).expanduser() if ssl_key_path else None

        if self.use_https and self.domain is None:
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
            logger.debug(f"Configuring Caddy for address: {self.address}")
            self._install()
            self._build_template()
            self._start_caddy()

        # Reload Caddy with the updated config
        logger.debug("Reloading Caddy")
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
            if SYSTEMCTL_ERROR in result.stderr:
                # If running in a docker container or distro without systemctl, reload caddy as a background process
                reload_cmd = f"caddy reload --config {str(self.caddyfile)}"
                try:
                    subprocess.run(reload_cmd, shell=True, check=True, text=True)
                except subprocess.CalledProcessError as e:
                    raise e
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
        if result.returncode == 0:
            logger.debug("Caddy is already installed, skipping install.")
        else:
            # Install caddy as a service (or background process if we can't use systemctl)
            # https://caddyserver.com/docs/running#using-the-service
            logger.info("Installing Caddy...")

            caddy_version = "2.7.6"
            arch = subprocess.run(
                "dpkg --print-architecture",
                shell=True,
                text=True,
                check=True,
                capture_output=True,
            ).stdout.strip()
            caddy_deb_url = f"https://github.com/caddyserver/caddy/releases/download/v{caddy_version}/caddy_{caddy_version}_linux_{arch}.deb"

            commands = [
                f"sudo wget -q {caddy_deb_url}",
                f'sudo apt install "./caddy_{caddy_version}_linux_{arch}.deb"',
            ]

            for cmd in commands:
                try:
                    subprocess.run(
                        cmd,
                        shell=True,
                        check=True,
                        text=True,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                except subprocess.CalledProcessError as e:
                    raise e

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
            # If custom certs provided use them instead of having Caddy generate them
            # https://caddyserver.com/docs/caddyfile/directives/tls
            logger.info("Using custom certs to enable HTTPs")
            tls_directive = f"tls {self.ssl_cert_path} {self.ssl_key_path}"
            # If domain also provided use it
            address_or_domain = self.domain or self.address
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
        elif self.domain:
            # https://caddyserver.com/docs/automatic-https#hostname-requirements
            logger.info(
                f"Generating certs with Caddy to enable HTTPs using domain: {self.domain}"
            )
            return textwrap.dedent(
                f"""
                https://{self.domain} {{
                    reverse_proxy 127.0.0.1:{self.rh_server_port}
                }}
                """
            ).strip()
        else:
            # Do not support issuing self-signed certs on the cluster
            # Unverified certs should be generated client side and passed to Caddy as custom certs
            raise RuntimeError("No certs or domain provided. Cannot enable HTTPS.")

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
        logger.debug("Checking Caddy configuration.")
        result = subprocess.run(
            ["sudo", "systemctl", "status", "caddy"],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            if SYSTEMCTL_ERROR in result.stderr:
                # If running in a docker container or distro without systemctl, check whether Caddy has been configured
                run_cmd = f"caddy validate --config {str(self.caddyfile)}"
                try:
                    subprocess.run(
                        run_cmd,
                        shell=True,
                        check=True,
                        text=True,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                except subprocess.CalledProcessError:
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
        logger.debug("Adding Caddy as trusted app.")
        try:
            subprocess.run(
                "sudo mkdir -p /var/lib/caddy/.local && "
                "sudo chown -R caddy: /var/lib/caddy",
                shell=True,
                check=True,
                text=True,
                stdout=subprocess.DEVNULL,
            )
        except subprocess.CalledProcessError as e:
            raise e

        logger.debug("Starting Caddy.")
        run_cmd = ["sudo", "systemctl", "start", "caddy"]
        result = subprocess.run(
            run_cmd,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            if SYSTEMCTL_ERROR in result.stderr:
                # If running in a docker container or distro without systemctl, we need to start Caddy manually
                # as a background process
                run_cmd = "caddy start"
                try:
                    subprocess.run(
                        run_cmd,
                        shell=True,
                        check=True,
                        text=True,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                except subprocess.CalledProcessError as e:
                    raise e
            else:
                raise RuntimeError(f"Failed to run Caddy service: {result.stderr}")

        logger.info("Successfully started Caddy.")
