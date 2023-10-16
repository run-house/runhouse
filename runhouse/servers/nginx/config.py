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

    def __init__(
        self,
        app_port: int,
        ssl_cert_path: str,
        ssl_key_path: str,
        force_reinstall=False,
        address: str = None,
    ):
        self.app_port = app_port
        self.ssl_cert_path = ssl_cert_path
        self.ssl_key_path = ssl_key_path
        self.force_reinstall = force_reinstall

        # To expose the server to the internet, set address to the public IP, otherwise leave it as localhost
        self.address = address or "localhost"

    def configure(self):
        """Configure nginx to proxy requests to the Fast API HTTP server"""
        self.install()

        is_configured = self._is_configured()
        if not self.force_reinstall and is_configured:
            logger.info("Nginx is already configured")
            return

        # Configure initial nginx settings
        self._apply_config()

        # Reload nginx with the updated config
        self.reload()

        if not self._is_configured():
            raise RuntimeError("Failed to configure Nginx")

    def reload(self):
        logger.info("Reloading nginx to apply changes.")
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
            logger.info("Installing nginx on server")
            result = subprocess.run(
                "sudo apt update && sudo apt install nginx -y",
                shell=True,
                check=True,
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                raise RuntimeError(f"Failed to install nginx: {result.stderr}")

    def _create_template(self):
        nginx_template = textwrap.dedent(
            """
        server {{
            listen 80;
            listen [::]:80;

            location / {{
                proxy_pass {proxy_pass};
                proxy_set_header Host $host;
                proxy_set_header X-Real-IP $remote_addr;
                proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            }}
        }}

        server {{
            listen 443 ssl default_server;
            listen [::]:443 ssl default_server;

            ssl_certificate {ssl_cert_path};
            ssl_certificate_key {ssl_key_path};
            ssl_protocols TLSv1 TLSv1.1 TLSv1.2;
            ssl_ciphers HIGH:!aNULL:!MD5;

            location / {{
                proxy_pass {proxy_pass};
                proxy_set_header Host $host;
                proxy_set_header X-Real-IP $remote_addr;
                proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            }}
        }}
        """
        )

        replace_dict = {
            "proxy_pass": f"http://{self.address}:{self.app_port}",
            "ssl_cert_path": self.ssl_cert_path,
            "ssl_key_path": self.ssl_key_path,
        }

        # Replace placeholders with actual values
        return nginx_template.format(**replace_dict)

    def _build_template(self):
        logger.info("Building Nginx template for the Runhouse API server.")
        subprocess.run(
            "sudo chmod o+w /etc/nginx/sites-available",
            shell=True,
            check=True,
            capture_output=True,
            text=True,
        )
        try:
            template = self._create_template()
            with open(self.BASE_CONFIG_PATH, "w") as f:
                f.write(template)

        except Exception as e:
            raise RuntimeError(f"Error configuring new Nginx template: {e}")

    def _is_configured(self) -> bool:
        if not Path(self.BASE_CONFIG_PATH).exists():
            self._build_template()

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

        if Path("/etc/nginx/sites-enabled/fastapi").exists():
            # Symlink already exists
            return

        result = subprocess.run(
            "sudo ln -s /etc/nginx/sites-available/fastapi /etc/nginx/sites-enabled && "
            "sudo ufw allow 'Nginx Full'",
            check=True,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Error configuring nginx: {result.stderr}")
