import textwrap
import unittest
from unittest.mock import MagicMock, mock_open, patch

from runhouse.servers.nginx.config import NginxConfig


class TestNginxConfiguration(unittest.TestCase):
    def setUp(self):
        self.config = NginxConfig(
            address="127.0.0.1",
            rh_server_port=32300,
            http_port=80,
            ssl_key_path="/path/to/ssl.key",
            ssl_cert_path="/path/to/ssl.cert",
            force_reinstall=False,
        )

    @patch("builtins.open", new_callable=mock_open)
    @patch("pathlib.Path.exists")
    @patch("subprocess.run")
    def test_nginx_config_generation(
        self, mock_subprocess_run, mock_path_exists, mock_file_open
    ):
        # Assume the paths do not exist for this test
        mock_path_exists.return_value = False

        # Mock the subprocess.run to pretend the chmod command succeeds
        mock_subprocess_run.return_value = MagicMock(returncode=0)

        self.config._build_template()

        mock_file_open.assert_called_once_with(self.config.BASE_CONFIG_PATH, "w")

        file_handle = mock_file_open()

        # Assert that we wrote the template to the file
        template = (
            self.config._https_template()
            if self.config._use_https
            else self.config._http_template()
        )
        file_handle.write.assert_called_once_with(template)

        # Assert that subprocess.run was called to change permissions
        mock_subprocess_run.assert_called()

    @patch("subprocess.run")
    def test_nginx_reload(self, mock_subprocess_run):
        # Mock the subprocess.run to pretend the reload command succeeds
        mock_subprocess_run.return_value = MagicMock(returncode=0)

        self.config.reload()

        # Assert that the subprocess.run was called with the nginx reload command
        mock_subprocess_run.assert_called_once_with(
            "sudo systemctl reload nginx",
            shell=True,
            check=True,
            capture_output=True,
            text=True,
        )

    def test_error_when_no_ssl_paths_provided_for_https(self):
        with self.assertRaises(ValueError) as context:
            NginxConfig(
                address="127.0.0.1",
                rh_server_port=32300,
                https_port=443,
                force_reinstall=False,
            )

        self.assertEqual(
            str(context.exception),
            "Must provide SSL certificate and key paths when using HTTPS",
        )

    @patch("builtins.open", new_callable=mock_open)
    @patch("pathlib.Path.exists")
    @patch("subprocess.run")
    def test_nginx_http_config_generation(
        self, mock_subprocess_run, mock_path_exists, mock_file_open
    ):
        # Set up NginxConfig for HTTP only
        config = NginxConfig(
            address="127.0.0.1",
            rh_server_port=32300,
            http_port=80,
            force_reinstall=False,
        )

        mock_path_exists.return_value = False

        mock_subprocess_run.return_value = MagicMock(returncode=0)

        config._build_template()

        # Assert that the file was opened for writing the HTTP template
        mock_file_open.assert_called_once_with(config.BASE_CONFIG_PATH, "w")

        # Retrieve the file handle from the mock
        file_handle = mock_file_open()

        # Assert that we wrote the HTTP template to the file
        http_template = config._http_template()
        file_handle.write.assert_called_once_with(http_template)

        # Check the calls to subprocess.run and ensure no SSL related commands were issued
        calls = mock_subprocess_run.call_args_list
        ssl_related_commands = [
            call_args
            for call_args in calls
            if "chmod" in call_args[0][0]
            and (
                "ssl_cert_path" in call_args[0][0] or "ssl_key_path" in call_args[0][0]
            )
        ]
        self.assertEqual(
            len(ssl_related_commands),
            0,
            "SSL related chmod commands should not be called for HTTP configuration",
        )

        self.assertFalse(config._use_https)
        self.assertIsNotNone(config.http_port)
        self.assertIsNone(config.https_port)
        self.assertIsNone(config.ssl_cert_path)
        self.assertIsNone(config.ssl_key_path)

        expected_http_template = textwrap.dedent(
            f"""
            server {{
                listen {config.http_port};

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

        self.assertEqual(http_template, expected_http_template)

    @patch("builtins.open", new_callable=mock_open)
    @patch("pathlib.Path.exists")
    @patch("subprocess.run")
    def test_nginx_https_config_generation(
        self, mock_subprocess_run, mock_path_exists, mock_file_open
    ):
        # Set up NginxConfig for HTTPS only
        config = NginxConfig(
            address="127.0.0.1",
            rh_server_port=32300,
            https_port=443,
            ssl_cert_path="/path/to/ssl.cert",
            ssl_key_path="/path/to/ssl.key",
            force_reinstall=False,
        )

        # Assume the paths do not exist for this test
        mock_path_exists.return_value = False

        mock_subprocess_run.return_value = MagicMock(returncode=0)

        config._build_template()

        # Assert that the file was opened for writing the HTTPS template
        mock_file_open.assert_called_once_with(config.BASE_CONFIG_PATH, "w")

        # Retrieve the file handle from the mock
        file_handle = mock_file_open()

        https_template = config._https_template()
        file_handle.write.assert_called_once_with(https_template)

        # Check the calls to subprocess.run and ensure no SSL related commands were issued
        calls = mock_subprocess_run.call_args_list
        ssl_related_commands = [
            call_args
            for call_args in calls
            if "chmod" in call_args[0][0]
            and (
                "ssl_cert_path" in call_args[0][0] or "ssl_key_path" in call_args[0][0]
            )
        ]
        self.assertEqual(
            len(ssl_related_commands),
            0,
            "SSL related chmod commands should not be called for HTTPS configuration",
        )

        self.assertTrue(config._use_https)
        self.assertIsNotNone(config.https_port)
        self.assertIsNone(config.http_port)
        self.assertIsNotNone(config.ssl_cert_path)
        self.assertIsNotNone(config.ssl_key_path)

        expected_https_template = textwrap.dedent(
            f"""
            server {{
                listen {config.https_port} ssl;

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

        self.assertEqual(https_template, expected_https_template)


if __name__ == "__main__":
    unittest.main()
