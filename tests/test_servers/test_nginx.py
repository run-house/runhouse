import textwrap
from pathlib import Path

import pytest
import requests

from runhouse.globals import rns_client
from runhouse.servers.http.http_utils import pickle_b64, PutObjectParams
from runhouse.servers.nginx.config import NginxConfig


@pytest.mark.servertest
class TestNginxConfiguration:
    @pytest.fixture(autouse=True)
    def init_fixtures(self, mocker):
        # mock existence of cert and key files
        mocker.patch.object(Path, "exists", return_value=True)

        # Create instances of NginxConfig with mocked existence of cert and key files
        self.http_config = NginxConfig(
            address="127.0.0.1",
        )

        self.https_config = NginxConfig(
            address="127.0.0.1",
            ssl_key_path="/path/to/ssl.key",
            ssl_cert_path="/path/to/ssl.cert",
            use_https=True,
        )

    @pytest.mark.level("unit")
    def test_nginx_http_build_config(self, mocker):
        # set up mocks
        mock_subprocess_run = mocker.patch("subprocess.run")
        mock_path_exists = mocker.patch("pathlib.Path.exists")
        mock_file_open = mocker.patch("builtins.open", new_callable=mocker.mock_open)

        # Assume the paths do not exist for this test
        mock_path_exists.return_value = False

        mock_subprocess_run.return_value = mocker.MagicMock(returncode=0)

        self.http_config._build_template()

        mock_file_open.assert_called_once_with(self.http_config.BASE_CONFIG_PATH, "w")

        file_handle = mock_file_open()

        template = self.http_config._http_template()
        file_handle.write.assert_called_once_with(template)

        # Assert that subprocess.run was called to change permissions
        mock_subprocess_run.assert_called()

    @pytest.mark.level("unit")
    def test_nginx_https_build_config(self, mocker):

        # set up mocks
        mock_subprocess_run = mocker.patch("subprocess.run")
        mock_path_exists = mocker.patch("pathlib.Path.exists")
        mock_file_open = mocker.patch("builtins.open", new_callable=mocker.mock_open)

        # Assume the paths do not exist for this test
        mock_path_exists.return_value = False

        mock_subprocess_run.return_value = mocker.MagicMock(returncode=0)

        self.https_config._build_template()

        mock_file_open.assert_called_once_with(self.https_config.BASE_CONFIG_PATH, "w")

        file_handle = mock_file_open()

        template = self.https_config._https_template()
        file_handle.write.assert_called_once_with(template)

        mock_subprocess_run.assert_called()

    @pytest.mark.level("unit")
    def test_nginx_http_reload(self, mocker):

        # set up mocks
        mock_subprocess_run = mocker.patch("subprocess.run")

        mock_subprocess_run.return_value = mocker.MagicMock(returncode=0)

        # Assuming self.http_config.address is set to a value like "localhost"
        expected_command = (
            "sudo service nginx start && sudo nginx -s reload"
            if self.http_config.address in ["localhost", "127.0.0.1", "0.0.0.0"]
            else "sudo systemctl reload nginx"
        )

        self.http_config.reload()

        mock_subprocess_run.assert_called_once_with(
            expected_command,
            shell=True,
            check=True,
            capture_output=True,
            text=True,
        )

    @pytest.mark.level("unit")
    def test_nginx_https_reload(self, mocker):

        # set up mocks
        mock_subprocess_run = mocker.patch("subprocess.run")

        mock_subprocess_run.return_value = mocker.MagicMock(returncode=0)

        # Assuming self.http_config.address is set to a value like "localhost"
        expected_command = (
            "sudo service nginx start && sudo nginx -s reload"
            if self.https_config.address in ["localhost", "127.0.0.1", "0.0.0.0"]
            else "sudo systemctl reload nginx"
        )

        self.https_config.reload()

        mock_subprocess_run.assert_called_once_with(
            expected_command,
            shell=True,
            check=True,
            capture_output=True,
            text=True,
        )

    @pytest.mark.level("unit")
    def test_nginx_reload_error_handling(self, mocker):

        # set up mocks
        mock_subprocess_run = mocker.patch("subprocess.run")

        # Simulate a failed reload
        mock_subprocess_run.return_value = mocker.MagicMock(
            returncode=1, stderr="Failed to reload"
        )

        with pytest.raises(RuntimeError):
            self.http_config.reload()

    @pytest.mark.level("unit")
    def test_http_firewall_rule_application(self, mocker):
        # set up mocks
        mock_subprocess_run = mocker.patch("subprocess.run")
        mock_path_exists = mocker.patch("pathlib.Path.exists")

        mock_subprocess_run.return_value = mocker.MagicMock(returncode=0)

        mock_path_exists.return_value = True

        self.http_config._apply_config()

        mock_subprocess_run.assert_any_call(
            "sudo ufw allow 'Nginx HTTP'",
            shell=True,
            check=True,
            capture_output=True,
            text=True,
        )

    @pytest.mark.level("unit")
    def test_https_firewall_rule_application(self, mocker):
        # set up mocks
        mock_subprocess_run = mocker.patch("subprocess.run")
        mock_path_exists = mocker.patch("pathlib.Path.exists")

        mock_subprocess_run.return_value = mocker.MagicMock(returncode=0)

        mock_path_exists.return_value = True

        self.https_config._apply_config()

        mock_subprocess_run.assert_any_call(
            "sudo ufw allow 'Nginx HTTPS'",
            shell=True,
            check=True,
            capture_output=True,
            text=True,
        )

    @pytest.mark.level("unit")
    def test_symlink_creation(self, mocker):

        # set up mocks
        mock_subprocess_run = mocker.patch("subprocess.run")
        mock_path_exists = mocker.patch("pathlib.Path.exists")

        # Simulate that the config file exists but the symlink does not
        mock_path_exists.side_effect = [
            True,
            False,
        ]  # First call returns True, second call returns False

        mock_subprocess_run.return_value = mocker.MagicMock(returncode=0)

        self.https_config._apply_config()

        mock_subprocess_run.assert_any_call(
            "sudo ln -s /etc/nginx/sites-available/fastapi /etc/nginx/sites-enabled",
            shell=True,
            check=True,
            capture_output=True,
            text=True,
        )

    @pytest.mark.level("unit")
    def test_invalid_ssl_paths_for_https(self, mocker):
        mocker.patch.object(Path, "exists", return_value=False)
        with pytest.raises(FileNotFoundError):
            # Incorrect SSL key and cert paths provided
            NginxConfig(
                address="127.0.0.1",
                ssl_key_path="/path/to/nonexistent/key",
                ssl_cert_path="/path/to/nonexistent/cert",
                use_https=True,
            )

    @pytest.mark.level("unit")
    def test_empty_ssl_paths_for_https(self):
        with pytest.raises(FileNotFoundError):
            # SSL key and cert paths not provided
            NginxConfig(
                address="127.0.0.1",
                use_https=True,
            )

    @pytest.mark.level("unit")
    def test_ignore_invalid_ssl_paths_for_http(self):
        # Invalid SSL key and cert paths not provided, which we ignore for HTTP config
        NginxConfig(
            address="127.0.0.1",
            ssl_cert_path="/path/to/nonexistent/cert",
            ssl_key_path="/path/to/nonexistent/key",
        )

    @pytest.mark.level("unit")
    def test_template_generation_based_on_http_config(self, mocker):

        # set up mocks
        mock_subprocess_run = mocker.patch("subprocess.run")
        mock_path_exists = mocker.patch("pathlib.Path.exists")
        mock_file_open = mocker.patch("builtins.open", new_callable=mocker.mock_open)

        mock_path_exists.return_value = False
        mock_subprocess_run.return_value = mocker.MagicMock(returncode=0)

        self.http_config._build_template()

        file_handle = mock_file_open()
        expected_template = self.http_config._http_template()
        file_handle.write.assert_called_once_with(expected_template)

    @pytest.mark.level("unit")
    def test_template_generation_based_on_https_config(self, mocker):

        # set up mocks
        mock_subprocess_run = mocker.patch("subprocess.run")
        mock_path_exists = mocker.patch("pathlib.Path.exists")
        mock_file_open = mocker.patch("builtins.open", new_callable=mocker.mock_open)

        mock_path_exists.return_value = False
        mock_subprocess_run.return_value = mocker.MagicMock(returncode=0)

        self.https_config._build_template()

        file_handle = mock_file_open()
        expected_template = self.https_config._https_template()
        file_handle.write.assert_called_once_with(expected_template)

    @pytest.mark.level("unit")
    def test_handling_different_addresses(self, mocker):

        # set up mocks
        mock_subprocess_run = mocker.patch("subprocess.run")
        mock_path_exists = mocker.patch("pathlib.Path.exists")
        mock_file_open = mocker.patch("builtins.open", new_callable=mocker.mock_open)

        mock_path_exists.return_value = False

        mock_subprocess_run.return_value = mocker.MagicMock(returncode=0)

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

    @pytest.mark.level("unit")
    def test_nginx_http_config_generation(self, mocker):

        mock_subprocess_run = mocker.patch("subprocess.run")
        mock_path_exists = mocker.patch("pathlib.Path.exists")
        mock_file_open = mocker.patch("builtins.open", new_callable=mocker.mock_open)

        # Set up NginxConfig for HTTP only
        config = NginxConfig(address="127.0.0.1")

        mock_path_exists.return_value = False

        mock_subprocess_run.return_value = mocker.MagicMock(returncode=0)

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
                    proxy_buffering off;
                    proxy_pass http://127.0.0.1:{config.rh_server_port}/;
                    send_timeout 3600;
                }}
            }}
            """
        )

        assert http_template == expected_http_template

    @pytest.mark.level("unit")
    def test_nginx_https_config_generation(self, mocker):

        # set up mocks
        mock_subprocess_run = mocker.patch("subprocess.run")
        mock_path_exists = mocker.patch("pathlib.Path.exists")
        mock_file_open = mocker.patch("builtins.open", new_callable=mocker.mock_open)

        # Set up NginxConfig for HTTPS only
        config = NginxConfig(
            address="127.0.0.1",
            ssl_cert_path="/path/to/ssl.cert",
            ssl_key_path="/path/to/ssl.key",
            use_https=True,
        )

        mock_path_exists.return_value = False

        mock_subprocess_run.return_value = mocker.MagicMock(returncode=0)

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
                    proxy_buffering off;
                    proxy_pass http://127.0.0.1:{config.rh_server_port}/;
                    send_timeout 3600;
                }}
            }}
            """
        )

        assert https_template == expected_https_template


@pytest.mark.servertest
class TestNginxServerLocally:

    UNIT = {
        "cluster": [
            "docker_cluster_pk_http_exposed",
            "docker_cluster_pk_tls_den_auth",
        ]
    }
    LOCAL = {
        "cluster": [
            "docker_cluster_pk_http_exposed",
            "docker_cluster_pk_tls_den_auth",
        ]
    }
    MINIMAL = {
        "cluster": [
            "docker_cluster_pk_http_exposed",
            "docker_cluster_pk_tls_den_auth",
        ]
    }
    THOROUGH = {
        "cluster": [
            "docker_cluster_pk_http_exposed",
            "docker_cluster_pk_tls_den_auth",
        ]
    }
    MAXIMAL = {
        "cluster": [
            "docker_cluster_pk_http_exposed",
            "docker_cluster_pk_tls_den_auth",
        ]
    }

    @pytest.mark.level("local")
    def test_using_nginx_on_local_cluster(self, cluster):
        protocol = "https" if cluster._use_https else "http"

        cluster.check_server()

        assert cluster.is_up()

        key = "key1"
        test_list = list(range(5, 50, 2)) + ["a string"]
        response = requests.post(
            f"{protocol}://{cluster.address}:{cluster.client_port}/object",
            json=PutObjectParams(
                serialized_data=pickle_b64(test_list), key=key, serialization="pickle"
            ).dict(),
            headers=rns_client.request_headers(),
            verify=False,
        )
        assert response.status_code == 200

        response = requests.get(
            f"{protocol}://{cluster.address}:{cluster.client_port}/keys",
            headers=rns_client.request_headers(),
            verify=False,
        )

        assert response.status_code == 200
        assert key in response.json().get("data")
