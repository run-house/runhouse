import logging
import subprocess
import textwrap
from pathlib import Path

logger = logging.getLogger(__name__)


class NginxConfig:
    BASE_CONFIG_PATH = "/etc/nginx/sites-available/fastapi"

    # Helpful commands:
    # sudo systemctl restart nginx
    # sudo systemctl status nginx

    # sudo apt-get install net-tools
    # sudo netstat -tuln | grep -E '80|443'

    def __init__(
        self,
        address: str,
        rh_server_port: int,
        http_port: int = None,
        https_port: int = None,
        ssl_cert_path: str = None,
        ssl_key_path: str = None,
        force_reinstall=False,
    ):
        self._use_https = https_port is not None
        self.http_port = http_port
        self.https_port = https_port
        self.rh_server_port = rh_server_port

        self.ssl_cert_path = ssl_cert_path
        self.ssl_key_path = ssl_key_path

        if self._use_https and (not self.ssl_cert_path or not self.ssl_key_path):
            raise ValueError(
                "Must provide SSL certificate and key paths when using HTTPS"
            )

        self.force_reinstall = force_reinstall

        # To expose the server to the internet, set address to the public IP, otherwise leave it as localhost
        self.address = address or "localhost"

    def configure(self):
        """Configure nginx to proxy requests to the Fast API HTTP server"""
        self.install()

        is_configured = self._is_configured()
        self._apply_config()

        if not self.force_reinstall and is_configured:
            logger.info("Nginx is already configured")
            return

        # Reload nginx with the updated config
        self.reload()

        if not self._is_configured():
            raise RuntimeError("Failed to configure Nginx")

    def reload(self):
        result = subprocess.run(
            "sudo systemctl reload nginx",
            shell=True,
            check=True,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Failed to reload Nginx: {result.stderr}")

    def install(self):
        try:
            subprocess.run(
                "sudo nginx -v", shell=True, check=True, capture_output=True, text=True
            )
        except subprocess.CalledProcessError:
            logger.info("Installing Nginx on server")
            result = subprocess.run(
                "sudo apt update && sudo apt install nginx -y",
                shell=True,
                check=True,
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                raise RuntimeError(f"Failed to install nginx: {result.stderr}")

    def _http_template(self):
        template = textwrap.dedent(
            """
        server {{
            listen {app_port};

            server_name {server_name};

            location / {{
                proxy_pass {proxy_pass};
                proxy_buffer_size 128k;
                proxy_buffers 4 256k;
                proxy_busy_buffers_size 256k;
            }}
        }}
        """
        )

        replace_dict = {
            "app_port": self.http_port,
            "proxy_pass": f"http://127.0.0.1:{self.rh_server_port}/",
            "server_name": self.address,
        }

        # Replace placeholders with actual values
        return template.format(**replace_dict)

    def _https_template(self):
        template = textwrap.dedent(
            """
        server {{
            listen {app_port} ssl;

            server_name {server_name};

            ssl_certificate {ssl_cert_path};
            ssl_certificate_key {ssl_key_path};

            location / {{
                proxy_pass {proxy_pass};
                proxy_buffer_size 128k;
                proxy_buffers 4 256k;
                proxy_busy_buffers_size 256k;
            }}
        }}
        """
        )

        replace_dict = {
            "app_port": self.https_port,
            "proxy_pass": f"http://127.0.0.1:{self.rh_server_port}/",
            "server_name": self.address,
            "ssl_cert_path": self.ssl_cert_path,
            "ssl_key_path": self.ssl_key_path,
        }

        # Replace placeholders with actual values
        return template.format(**replace_dict)

    def _build_template(self):
        if Path(self.BASE_CONFIG_PATH).exists():
            return

        if self._use_https:
            subprocess.run(
                f"sudo chmod 600 {self.ssl_cert_path} && "
                f"sudo chmod 600 {self.ssl_key_path}",
                shell=True,
                check=True,
                capture_output=True,
                text=True,
            )

        subprocess.run(
            "sudo chmod o+w /etc/nginx/sites-available",
            shell=True,
            check=True,
            capture_output=True,
            text=True,
        )
        try:
            template = (
                self._https_template() if self._use_https else self._http_template()
            )
            with open(self.BASE_CONFIG_PATH, "w") as f:
                f.write(template)

        except Exception as e:
            raise RuntimeError(f"Error configuring new Nginx template: {e}")

    def _is_configured(self) -> bool:
        if not Path(self.BASE_CONFIG_PATH).exists():
            self._build_template()

        logger.info("Checking Nginx configuration.")
        result = subprocess.run(
            ["sudo", "nginx", "-t", "-c", "/etc/nginx/nginx.conf"],
            check=True,
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            logger.error(result.stderr)
            logger.error(
                "Nginx is not configured correctly, attempting to re-configure."
            )

        return "syntax is ok" in result.stderr and "test is successful" in result.stderr

    def _apply_config(self):
        logger.info("Applying Nginx settings.")
        self._build_template()

        # Allow incoming traffic to the default port (HTTPS or HTTP)
        # e.g. 'Nginx HTTPS': predefined profile that allows traffic for the Nginx web server on port 443
        ufw_rule = "Nginx HTTPS" if self._use_https else "Nginx HTTP"
        logger.info(f"Allowing incoming traffic using rule: `{ufw_rule}`")
        result = subprocess.run(
            f"sudo ufw allow '{ufw_rule}'",
            check=True,
            capture_output=True,
            text=True,
            shell=True,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"Error configuring firewall to open default ports: {result.stderr}"
            )

        if Path("/etc/nginx/sites-enabled/fastapi").exists():
            # Symlink already exists
            return

        result = subprocess.run(
            "sudo ln -s /etc/nginx/sites-available/fastapi /etc/nginx/sites-enabled",
            check=True,
            capture_output=True,
            text=True,
            shell=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Error configuring nginx: {result.stderr}")

        # Reload for the UFW firewall rules to take effect
        self.reload()
