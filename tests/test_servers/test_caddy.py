import tempfile
import textwrap
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import requests

from runhouse.globals import rns_client
from runhouse.servers.caddy.config import CaddyConfig
from runhouse.servers.http.http_utils import PutObjectParams, serialize_data


@pytest.mark.servertest
class TestCaddyConfiguration:
    @pytest.fixture(autouse=True)
    def init_fixtures(self):
        # mock existence of cert and key files
        with (patch.object(Path, "exists", return_value=True)):
            self.http_config = CaddyConfig(
                address="127.0.0.1",
            )

            # Using HTTPS client with Caddy handling certs generation
            self.https_config_caddy_certs = CaddyConfig(
                address="127.0.0.1", use_https=True, domain="run.house"
            )

            # Using temporary certs for testing
            with tempfile.NamedTemporaryFile(
                suffix=".cert", delete=False
            ) as cert_file, tempfile.NamedTemporaryFile(
                suffix=".key", delete=False
            ) as key_file:
                self.temp_cert_path = cert_file.name
                self.temp_key_path = key_file.name

            # Using HTTPS client with pre-existing custom certs
            self.https_config_custom_certs = CaddyConfig(
                address="127.0.0.1",
                use_https=True,
                ssl_key_path=self.temp_key_path,
                ssl_cert_path=self.temp_cert_path,
            )

            self.caddy_certs_template = self.https_config_caddy_certs._https_template()
            self.custom_certs_template = (
                self.https_config_custom_certs._https_template()
            )

    # -----------------------------------------

    @pytest.mark.level("unit")
    @patch("subprocess.run")
    def test_caddy_http_build_config(self, mock_subprocess_run):
        mock_subprocess_run.return_value = MagicMock(returncode=0)

        self.http_config._build_template()

        template = self.http_config._http_template()
        assert f"http://{self.http_config.address}" in template

        # Assert that subprocess.run was called to change permissions
        mock_subprocess_run.assert_called()

    @pytest.mark.level("unit")
    @patch("subprocess.run")
    def test_https_custom_certs_build_config(self, mock_subprocess_run):
        mock_subprocess_run.return_value = MagicMock(returncode=0)

        self.https_config_custom_certs._build_template()

        template = self.https_config_custom_certs._https_template()
        assert template == self.custom_certs_template

        # Assert that subprocess.run was called to change permissions
        mock_subprocess_run.assert_called()

    @pytest.mark.level("unit")
    @patch("subprocess.run")
    def test_https_caddy_certs_build_config(self, mock_subprocess_run):
        mock_subprocess_run.return_value = MagicMock(returncode=0)

        self.https_config_caddy_certs._build_template()

        template = self.https_config_caddy_certs._https_template()
        assert template == self.caddy_certs_template

        # Assert that subprocess.run was called to change permissions
        mock_subprocess_run.assert_called()

    @pytest.mark.level("unit")
    @patch("subprocess.run")
    def test_caddy_http_reload(self, mock_subprocess_run):
        mock_subprocess_run.return_value = MagicMock(returncode=0)

        expected_command = ["sudo", "systemctl", "reload", "caddy"]

        self.http_config.reload()

        mock_subprocess_run.assert_called_once_with(
            expected_command,
            capture_output=True,
            text=True,
        )

    @pytest.mark.level("unit")
    @patch("subprocess.run")
    def test_caddy_https_reload(self, mock_subprocess_run):
        mock_subprocess_run.return_value = MagicMock(returncode=0)

        expected_command = ["sudo", "systemctl", "reload", "caddy"]

        self.https_config_custom_certs.reload()

        mock_subprocess_run.assert_called_once_with(
            expected_command,
            capture_output=True,
            text=True,
        )

    @pytest.mark.level("unit")
    @patch("subprocess.run")
    def test_caddy_reload_error_handling(self, mock_subprocess_run):
        # Simulate a failed reload
        mock_subprocess_run.return_value = MagicMock(
            returncode=1, stderr="Failed to reload"
        )

        with pytest.raises(RuntimeError):
            self.http_config.reload()

    @pytest.mark.level("unit")
    @patch("pathlib.Path.exists")
    @patch("subprocess.run")
    def test_https_firewall_rule_application(
        self, mock_subprocess_run, mock_path_exists
    ):
        mock_subprocess_run.return_value = MagicMock(returncode=0)

        mock_path_exists.return_value = True

        self.https_config_caddy_certs._build_template()

        mock_subprocess_run.assert_any_call(
            "sudo ufw allow 443/tcp",
            shell=True,
            check=True,
            capture_output=True,
            text=True,
        )

    @pytest.mark.level("unit")
    def test_invalid_ssl_paths_for_https(self):
        with pytest.raises(FileNotFoundError):
            # Incorrect SSL key and cert paths provided
            CaddyConfig(
                address="127.0.0.1",
                ssl_cert_path="/path/to/nonexistent/cert",
                ssl_key_path="/path/to/nonexistent/key",
                use_https=True,
            )

    @pytest.mark.level("unit")
    def test_ignore_invalid_ssl_paths_for_http(self):
        # Invalid SSL key and cert paths not provided, which we ignore for HTTP config
        CaddyConfig(
            address="127.0.0.1",
            ssl_cert_path="/path/to/nonexistent/cert",
            ssl_key_path="/path/to/nonexistent/key",
        )

    @pytest.mark.level("unit")
    @patch("subprocess.run")
    def test_template_generation_based_on_http_config(self, mock_subprocess_run):
        mock_subprocess_run.return_value = MagicMock(returncode=0)

        self.http_config._build_template()

        expected_template = self.http_config._http_template()
        assert expected_template == self.http_config._http_template()

    @pytest.mark.level("unit")
    @patch("subprocess.run")
    def test_template_generation_based_on_https_custom_certs_config(
        self, mock_subprocess_run
    ):
        mock_subprocess_run.return_value = MagicMock(returncode=0)

        self.https_config_custom_certs._build_template()

        expected_template = self.https_config_custom_certs._https_template()
        assert expected_template == self.custom_certs_template

    @pytest.mark.level("unit")
    @patch("subprocess.run")
    def test_template_generation_based_on_https_caddy_certs_config(
        self, mock_subprocess_run
    ):
        mock_subprocess_run.return_value = MagicMock(returncode=0)

        self.https_config_caddy_certs._build_template()

        expected_template = self.https_config_caddy_certs._https_template()
        assert expected_template == self.caddy_certs_template

    @pytest.mark.level("unit")
    @patch("subprocess.run")
    def test_caddy_http_config_generation(self, mock_subprocess_run):
        # Set up CaddyConfig for HTTP only
        config = CaddyConfig(address="127.0.0.1")

        mock_subprocess_run.return_value = MagicMock(returncode=0)

        config._build_template()

        # Assert that we wrote the HTTP template to the file
        http_template = config._http_template()

        assert not config.use_https
        assert config.ssl_cert_path is None
        assert config.ssl_key_path is None

        expected_http_template = textwrap.dedent(
            f"""
            {{
                default_sni {config.address}
            }}

            http://{config.address} {{
                reverse_proxy localhost:{config.rh_server_port}
            }}
        """
        ).strip()

        assert http_template == expected_http_template

    @pytest.mark.level("unit")
    @patch("subprocess.run")
    def test_caddy_https_custom_certs_config_generation(self, mock_subprocess_run):
        # Set up CaddyConfig for HTTPS only
        config = CaddyConfig(
            address="127.0.0.1",
            ssl_cert_path=self.temp_cert_path,
            ssl_key_path=self.temp_key_path,
            use_https=True,
        )

        mock_subprocess_run.return_value = MagicMock(returncode=0)

        config._build_template()

        https_template = config._https_template()

        assert config.use_https
        assert config.ssl_cert_path is not None
        assert config.ssl_key_path is not None

        assert https_template == self.custom_certs_template

    @pytest.mark.level("unit")
    @patch("subprocess.run")
    def test_caddy_https_caddy_certs_config_generation(self, mock_subprocess_run):
        # Set up CaddyConfig for HTTPS only
        config = CaddyConfig(address="127.0.0.1", use_https=True, domain="run.house")

        mock_subprocess_run.return_value = MagicMock(returncode=0)

        config._build_template()

        https_template = config._https_template()

        assert config.use_https

        # Caddy will handle cert creation automatically
        assert config.ssl_cert_path is None
        assert config.ssl_key_path is None

        assert https_template == self.caddy_certs_template

    @pytest.mark.level("unit")
    def test_invalid_https_configuration(self):
        # Set up CaddyConfig for HTTPS without specifying valid cert files or a domain
        with pytest.raises(ValueError):
            CaddyConfig(address="127.0.0.1", use_https=True)

        with pytest.raises(FileNotFoundError):
            CaddyConfig(
                address="127.0.0.1",
                use_https=True,
                ssl_cert_path="/some/random/path",
                ssl_key_path="/some/random/other/path",
            )

    @pytest.mark.level("unit")
    def test_use_certs_even_if_domain_provided(self):
        cc = CaddyConfig(
            address="127.0.0.1",
            use_https=True,
            ssl_cert_path="/some/random/path",
            ssl_key_path="/some/random/other/path",
            domain="run.house",
        )
        assert (
            str(cc.ssl_cert_path) == "/some/random/path"
            and str(cc.ssl_key_path) == "/some/random/other/path"
        )
        assert f"tls {cc.ssl_cert_path} {cc.ssl_key_path}" in cc._https_template()

    @pytest.mark.level("unit")
    def test_use_domain_with_no_certs(self):
        cc = CaddyConfig(address="127.0.0.1", use_https=True, domain="run.house")
        assert cc.ssl_cert_path is None and cc.ssl_key_path is None
        assert "https://run.house" in cc._https_template()


@pytest.mark.servertest
class TestCaddyServerLocally:
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
    RELEASE = {
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
    def test_using_caddy_on_local_cluster(self, cluster):
        protocol = "https" if cluster._use_https else "http"

        cluster.client.check_server()

        assert cluster.is_up()

        key = "key1"
        test_list = list(range(5, 50, 2)) + ["a string"]
        verify = cluster.client.verify
        response = requests.post(
            f"{protocol}://{cluster.server_address}:{cluster.client_port}/object",
            json=PutObjectParams(
                serialized_data=serialize_data(test_list, "pickle"),
                key=key,
                serialization="pickle",
            ).model_dump(),
            headers=rns_client.request_headers(cluster.rns_address),
            verify=verify,
        )
        assert response.status_code == 200

        response = requests.get(
            f"{protocol}://{cluster.server_address}:{cluster.client_port}/keys",
            headers=rns_client.request_headers(cluster.rns_address),
            verify=verify,
        )

        assert response.status_code == 200
        assert key in response.json().get("data")
