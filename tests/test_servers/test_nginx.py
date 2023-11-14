import textwrap
import unittest
from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

import pytest

from runhouse.servers.nginx.config import NginxConfig


class TestNginxConfiguration:
    @pytest.fixture(autouse=True)
    def init_fixtures(self):
        # mock existence of cert and key files
        with patch.object(Path, "exists", return_value=True):
            self.http_config = NginxConfig(
                address="127.0.0.1",
            )
            self.https_config = NginxConfig(
                address="127.0.0.1",
                ssl_key_path="/path/to/ssl.key",
                ssl_cert_path="/path/to/ssl.cert",
                use_https=True,
            )

    @patch("builtins.open", new_callable=mock_open)
    @patch("pathlib.Path.exists")
    @patch("subprocess.run")
    def test_nginx_http_build_config(
        self, mock_subprocess_run, mock_path_exists, mock_file_open
    ):
        # Assume the paths do not exist for this test
        mock_path_exists.return_value = False

        mock_subprocess_run.return_value = MagicMock(returncode=0)

        self.http_config._build_template()

        mock_file_open.assert_called_once_with(self.http_config.BASE_CONFIG_PATH, "w")

        file_handle = mock_file_open()

        template = self.http_config._http_template()
        file_handle.write.assert_called_once_with(template)

        # Assert that subprocess.run was called to change permissions
        mock_subprocess_run.assert_called()

    @patch("builtins.open", new_callable=mock_open)
    @patch("pathlib.Path.exists")
    @patch("subprocess.run")
    def test_nginx_https_build_config(
        self, mock_subprocess_run, mock_path_exists, mock_file_open
    ):
        # Assume the paths do not exist for this test
        mock_path_exists.return_value = False

        mock_subprocess_run.return_value = MagicMock(returncode=0)

        self.https_config._build_template()

        mock_file_open.assert_called_once_with(self.https_config.BASE_CONFIG_PATH, "w")

        file_handle = mock_file_open()

        template = self.https_config._https_template()
        file_handle.write.assert_called_once_with(template)

        mock_subprocess_run.assert_called()

    @patch("subprocess.run")
    def test_nginx_http_reload(self, mock_subprocess_run):
        mock_subprocess_run.return_value = MagicMock(returncode=0)

        self.http_config.reload()

        mock_subprocess_run.assert_called_once_with(
            "sudo systemctl reload nginx",
            shell=True,
            check=True,
            capture_output=True,
            text=True,
        )

    @patch("subprocess.run")
    def test_nginx_https_reload(self, mock_subprocess_run):
        mock_subprocess_run.return_value = MagicMock(returncode=0)

        self.https_config.reload()

        mock_subprocess_run.assert_called_once_with(
            "sudo systemctl reload nginx",
            shell=True,
            check=True,
            capture_output=True,
            text=True,
        )

    @patch("subprocess.run")
    def test_nginx_reload_error_handling(self, mock_subprocess_run):
        # Simulate a failed reload
        mock_subprocess_run.return_value = MagicMock(
            returncode=1, stderr="Failed to reload"
        )

        with pytest.raises(RuntimeError):
            self.http_config.reload()

    @patch("pathlib.Path.exists")
    @patch("subprocess.run")
    def test_http_firewall_rule_application(
        self, mock_subprocess_run, mock_path_exists
    ):
        mock_subprocess_run.return_value = MagicMock(returncode=0)

        mock_path_exists.return_value = True

        self.http_config._apply_config()

        mock_subprocess_run.assert_any_call(
            "sudo ufw allow 'Nginx HTTP'",
            shell=True,
            check=True,
            capture_output=True,
            text=True,
        )

    @patch("pathlib.Path.exists")
    @patch("subprocess.run")
    def test_https_firewall_rule_application(
        self, mock_subprocess_run, mock_path_exists
    ):
        mock_subprocess_run.return_value = MagicMock(returncode=0)

        mock_path_exists.return_value = True

        self.https_config._apply_config()

        mock_subprocess_run.assert_any_call(
            "sudo ufw allow 'Nginx HTTPS'",
            shell=True,
            check=True,
            capture_output=True,
            text=True,
        )

    @patch("pathlib.Path.exists")
    @patch("subprocess.run")
    def test_symlink_creation(self, mock_subprocess_run, mock_path_exists):
        # Simulate that the config file exists but the symlink does not
        mock_path_exists.side_effect = [
            True,
            False,
        ]  # First call returns True, second call returns False

        mock_subprocess_run.return_value = MagicMock(returncode=0)

        self.https_config._apply_config()

        mock_subprocess_run.assert_any_call(
            "sudo ln -s /etc/nginx/sites-available/fastapi /etc/nginx/sites-enabled",
            shell=True,
            check=True,
            capture_output=True,
            text=True,
        )

    def test_invalid_ssl_paths_for_https(self):
        with pytest.raises(FileNotFoundError):
            # Incorrect SSL key and cert paths provided
            NginxConfig(
                address="127.0.0.1",
                ssl_cert_path="/path/to/nonexistent/cert",
                ssl_key_path="/path/to/nonexistent/key",
                use_https=True,
            )

    def test_empty_ssl_paths_for_https(self):
        with pytest.raises(FileNotFoundError):
            # SSL key and cert paths not provided
            NginxConfig(
                address="127.0.0.1",
                use_https=True,
            )

    def test_ignore_invalid_ssl_paths_for_http(self):
        # Invalid SSL key and cert paths not provided, which we ignore for HTTP config
        NginxConfig(
            address="127.0.0.1",
            ssl_cert_path="/path/to/nonexistent/cert",
            ssl_key_path="/path/to/nonexistent/key",
        )

    @patch("builtins.open", new_callable=mock_open)
    @patch("pathlib.Path.exists")
    @patch("subprocess.run")
    def test_template_generation_based_on_http_config(
        self, mock_subprocess_run, mock_path_exists, mock_file_open
    ):
        mock_path_exists.return_value = False
        mock_subprocess_run.return_value = MagicMock(returncode=0)

        self.http_config._build_template()

        file_handle = mock_file_open()
        expected_template = self.http_config._http_template()
        file_handle.write.assert_called_once_with(expected_template)

    @patch("builtins.open", new_callable=mock_open)
    @patch("pathlib.Path.exists")
    @patch("subprocess.run")
    def test_template_generation_based_on_https_config(
        self, mock_subprocess_run, mock_path_exists, mock_file_open
    ):
        mock_path_exists.return_value = False
        mock_subprocess_run.return_value = MagicMock(returncode=0)

        self.https_config._build_template()

        file_handle = mock_file_open()
        expected_template = self.https_config._https_template()
        file_handle.write.assert_called_once_with(expected_template)

    @patch("subprocess.run")
    @patch("builtins.open", new_callable=mock_open)
    @patch("pathlib.Path.exists")
    def test_handling_different_addresses(
        self, mock_path_exists, mock_file_open, mock_subprocess_run
    ):
        mock_path_exists.return_value = False

        mock_subprocess_run.return_value = MagicMock(returncode=0)

        # Test with localhost
        self.http_config.address = "localhost"
        self.http_config._build_template()
        localhost_template = self.http_config._http_template()
        mock_file_open().write.assert_called_with(localhost_template)

        # Test with public IP
        self.http_config.address = "192.168.1.1"
        self.http_config._build_template()
        public_ip_template = self.http_config._http_template()
        mock_file_open().write.assert_called_with(public_ip_template)

    @patch("builtins.open", new_callable=mock_open)
    @patch("pathlib.Path.exists")
    @patch("subprocess.run")
    def test_nginx_http_config_generation(
        self, mock_subprocess_run, mock_path_exists, mock_file_open
    ):
        # Set up NginxConfig for HTTP only
        config = NginxConfig(address="127.0.0.1")

        mock_path_exists.return_value = False

        mock_subprocess_run.return_value = MagicMock(returncode=0)

        config._build_template()

        # Assert that the file was opened for writing the HTTP template
        mock_file_open.assert_called_once_with(config.BASE_CONFIG_PATH, "w")

        file_handle = mock_file_open()

        # Assert that we wrote the HTTP template to the file
        http_template = config._http_template()
        file_handle.write.assert_called_once_with(http_template)

        calls = mock_subprocess_run.call_args_list
        ssl_related_commands = [
            call_args
            for call_args in calls
            if "chmod" in call_args[0][0]
            and (
                (
                    self.http_config.ssl_cert_path is not None
                    and self.http_config.ssl_cert_path in call_args[0][0]
                )
                or (
                    self.http_config.ssl_key_path is not None
                    and self.http_config.ssl_key_path in call_args[0][0]
                )
            )
        ]

        assert (
            len(ssl_related_commands) == 0
        ), "SSL related chmod commands should not be called for HTTP configuration"

        assert not config.use_https
        assert config.ssl_cert_path is None
        assert config.ssl_key_path is None

        expected_http_template = textwrap.dedent(
            f"""
            server {{
                listen 80;

                server_name {config.address};

                location / {{
                    proxy_pass http://127.0.0.1:{config.rh_server_port}/;
                    proxy_buffer_size 128k;
                    proxy_buffers 4 256k;
                    proxy_busy_buffers_size 256k;
                }}
            }}
            """
        )

        assert http_template == expected_http_template

    @patch("builtins.open", new_callable=mock_open)
    @patch("pathlib.Path.exists")
    @patch("subprocess.run")
    def test_nginx_https_config_generation(
        self, mock_subprocess_run, mock_path_exists, mock_file_open
    ):
        # Set up NginxConfig for HTTPS only
        config = NginxConfig(
            address="127.0.0.1",
            ssl_cert_path="/path/to/ssl.cert",
            ssl_key_path="/path/to/ssl.key",
            use_https=True,
        )

        mock_path_exists.return_value = False

        mock_subprocess_run.return_value = MagicMock(returncode=0)

        config._build_template()

        # Assert that the file was opened for writing the HTTPS template
        mock_file_open.assert_called_once_with(config.BASE_CONFIG_PATH, "w")

        file_handle = mock_file_open()

        https_template = config._https_template()
        file_handle.write.assert_called_once_with(https_template)

        calls = mock_subprocess_run.call_args_list
        ssl_related_commands = [
            call_args
            for call_args in calls
            if "chmod" in call_args[0][0]
            and (
                self.https_config.ssl_cert_path in call_args[0][0]
                or self.https_config.ssl_key_path in call_args[0][0]
            )
        ]
        assert (
            len(ssl_related_commands) == 1
        ), "SSL related chmod commands should be called for HTTPS configuration"

        assert config.use_https
        assert config.ssl_cert_path is not None
        assert config.ssl_key_path is not None

        expected_https_template = textwrap.dedent(
            f"""
            server {{
                listen 443 ssl;

                server_name {config.address};

                ssl_certificate {config.ssl_cert_path};
                ssl_certificate_key {config.ssl_key_path};

                location / {{
                    proxy_pass http://127.0.0.1:{config.rh_server_port}/;
                    proxy_buffer_size 128k;
                    proxy_buffers 4 256k;
                    proxy_busy_buffers_size 256k;
                }}
            }}
            """
        )

        assert https_template == expected_https_template


if __name__ == "__main__":
    unittest.main()
