import json
from unittest.mock import patch

import pytest

import runhouse as rh
from runhouse.constants import DEFAULT_SERVER_PORT, EMPTY_DEFAULT_ENV_NAME

from runhouse.globals import rns_client

from runhouse.servers.http import HTTPClient
from runhouse.servers.http.http_utils import (
    DeleteObjectParams,
    PutObjectParams,
    serialize_data,
)


@pytest.mark.servertest
class TestHTTPClient:
    @pytest.fixture(autouse=True)
    def init_fixtures(self):
        from tests.fixtures.secret_fixtures import provider_secret_values

        args = dict(
            name="local-cluster",
            host="localhost",
            server_host="0.0.0.0",
            ssh_creds=provider_secret_values["ssh"],
        )
        self.local_cluster = rh.cluster(**args)
        self.client = HTTPClient(
            "localhost",
            DEFAULT_SERVER_PORT,
            resource_address=self.local_cluster.rns_address,
        )

    @pytest.mark.level("unit")
    def test_check_server(self, mocker):
        mock_response = mocker.Mock()
        mock_response.status_code = 200
        rh_version_resp = {"rh_version": rh.__version__}
        mock_response.json.return_value = rh_version_resp
        mocked_get = mocker.patch("requests.Session.get", return_value=mock_response)

        self.client.check_server()
        expected_verify = self.client.verify

        mocked_get.assert_called_once_with(
            f"http://localhost:{DEFAULT_SERVER_PORT}/check",
            timeout=HTTPClient.CHECK_TIMEOUT_SEC,
            verify=expected_verify,
        )

    @pytest.mark.level("unit")
    def test_get_certificate(self, mocker):

        mock_request = mocker.patch("runhouse.servers.http.HTTPClient.request")
        mock_request.return_value = b"certificate_content"

        # Set up the mocker for 'mkdir' method
        mock_mkdir = mocker.patch("pathlib.Path.mkdir")

        # Set up the mocker for 'open' method
        mock_file_open = mocker.patch(
            "builtins.open",
            new_callable=mocker.mock_open,
            read_data="certificate_content",
        )

        self.client.cert_path = "/fake/path/cert.pem"

        self.client.get_certificate()

        mock_request.assert_called_once_with("cert", req_type="get", headers={})

        mock_mkdir.assert_called_once()

        # Check that open was called correctly
        mock_file_open.assert_called_once_with("/fake/path/cert.pem", "wb")

        # Check that the correct content was written to the file
        mock_file_open().write.assert_called_once_with(b"certificate_content")

    @pytest.mark.level("unit")
    @patch("runhouse.globals.rns_client.request_headers")
    def test_use_cert_verification(self, mock_request_headers, mocker):
        # Mock the request_headers to avoid actual HTTP requests in the test for loading the cluster token
        mock_request_headers.return_value = {"Authorization": "Bearer mock_token"}

        # Mock a certificate where the issuer is different from the subject
        mock_cert = mocker.MagicMock()
        mock_cert.issuer = "issuer"
        mock_cert.subject = "subject"

        mock_load_cert = mocker.patch(
            "builtins.open", mocker.mock_open(read_data="cert_data")
        )
        mocker.patch(
            "cryptography.x509.load_pem_x509_certificate", return_value=mock_cert
        )

        mock_load_cert.return_value = mock_cert

        # Test with HTTPS enabled and a valid cert path which is not self-signed
        client = HTTPClient(
            "localhost",
            DEFAULT_SERVER_PORT,
            resource_address=self.local_cluster.rns_address,
            use_https=True,
            cert_path="/valid/path",
        )
        assert client.verify is True

        # Mock a self-signed cert where the issuer is the same as the subject
        mock_cert.issuer = "self-signed"
        mock_cert.subject = "self-signed"
        mock_load_cert.return_value = mock_cert

        # If providing a valid self-signed cert, "verify" should be the path to the cert
        mocker.patch("pathlib.Path.exists", return_value=True)
        client = HTTPClient(
            "localhost",
            DEFAULT_SERVER_PORT,
            resource_address=self.local_cluster.rns_address,
            use_https=True,
            cert_path="/self-signed/path",
        )
        assert client.verify == "/self-signed/path"

        # If providing an invalid cert path we still default to verify=True since https is enabled
        mocker.patch("pathlib.Path.exists", return_value=False)
        client = HTTPClient(
            "localhost",
            DEFAULT_SERVER_PORT,
            resource_address=self.local_cluster.rns_address,
            use_https=True,
            cert_path="/invalid/path",
        )
        assert client.verify is True

    @pytest.mark.level("unit")
    def test_call_module_method(self, mocker):
        expected_headers = rns_client.request_headers(
            resource_address=self.local_cluster.rns_address
        )
        response_sequence = [
            json.dumps({"output_type": "stdout", "data": "Log message"}),
            json.dumps(
                {
                    "output_type": "result_serialized",
                    "data": serialize_data("final_result", "pickle"),
                    "serialization": "pickle",
                }
            ),
        ]

        # Mock the response to iter_lines to return our simulated server response
        mock_response = mocker.Mock()
        mock_response.status_code = 200
        mock_response.iter_lines.return_value = iter(response_sequence)
        mock_post = mocker.patch("requests.Session.post", return_value=mock_response)

        # Call the method under test
        method_name = "install"
        module_name = EMPTY_DEFAULT_ENV_NAME

        # Need to specify the run_name to avoid generating a unique one that contains the timestamp
        result = self.client.call(
            module_name,
            method_name,
            run_name="test_run_name",
        )

        assert result == "final_result"

        # Assert that the post request was called correctly
        expected_url = self.client._formatted_url(f"{module_name}/{method_name}")
        expected_json_data = {
            "data": None,
            "serialization": "pickle",
            "run_name": "test_run_name",
            "stream_logs": True,
            "save": False,
            "remote": False,
        }

        expected_verify = self.client.verify

        mock_post.assert_called_once_with(
            expected_url,
            json=expected_json_data,
            headers=expected_headers,
            auth=None,
            stream=True,
            verify=expected_verify,
        )

    @pytest.mark.level("unit")
    def test_call_module_method_with_args_kwargs(self, mocker):
        expected_headers = rns_client.request_headers(self.local_cluster.rns_address)

        mock_response = mocker.MagicMock()
        mock_response.status_code = 200
        # Set up iter_lines to return an iterator
        mock_response.iter_lines.return_value = iter(
            [
                json.dumps({"output_type": "log", "data": "Log message"}),
            ]
        )
        mock_post = mocker.patch("requests.Session.post", return_value=mock_response)

        data = {"args": [1, 2], "kwargs": {"a": 3, "b": 4}}
        module_name = "module"
        method_name = "install"

        # Need to specify the run_name to avoid generating a unique one that contains the timestamp
        self.client.call(
            module_name,
            method_name,
            data=data,
            run_name="test_run_name",
        )

        # Assert that the post request was called with the correct data
        expected_json_data = {
            "data": serialize_data(data, "pickle"),
            "serialization": "pickle",
            "run_name": "test_run_name",
            "stream_logs": True,
            "save": False,
            "remote": False,
        }
        expected_url = f"http://localhost:32300/{module_name}/{method_name}"
        expected_verify = self.client.verify

        mock_post.assert_called_with(
            expected_url,
            json=expected_json_data,
            headers=expected_headers,
            auth=None,
            stream=True,
            verify=expected_verify,
        )

    @pytest.mark.level("unit")
    def test_call_module_method_error_handling(self, mocker, local_cluster):
        mock_response = mocker.Mock()
        mock_response.status_code = 500
        mock_response.content = b"Internal Server Error"
        mocker.patch("requests.Session.post", return_value=mock_response)

        with pytest.raises(ValueError):
            self.client.call("module", "method")

    @pytest.mark.level("unit")
    def test_call_module_method_config(self, mocker, local_cluster):
        request_headers = rns_client.request_headers(local_cluster.rns_address)

        test_data = self.local_cluster.config()
        mock_response = mocker.Mock()
        mock_response.status_code = 200
        mock_response.iter_lines.return_value = iter(
            [
                json.dumps({"output_type": "config", "data": test_data}),
            ]
        )
        mocker.patch("requests.Session.post", return_value=mock_response)

        cluster = self.client.call(
            EMPTY_DEFAULT_ENV_NAME,
            "install",
            headers=request_headers,
        )
        assert cluster.config() == test_data

    @pytest.mark.level("unit")
    def test_put_object(self, mocker):

        mock_request = mocker.patch("runhouse.servers.http.HTTPClient.request_json")

        key = "my_list"
        value = list(range(5, 50, 2)) + ["a string"]
        expected_data = serialize_data(value, "pickle")

        self.client.put_object(key, value)

        mock_request.assert_called_once_with(
            "object",
            req_type="post",
            json_dict=mocker.ANY,
            err_str=f"Error putting object {key}",
        )

        actual_data = PutObjectParams(**mock_request.call_args[1]["json_dict"])
        assert actual_data.key == key
        assert actual_data.serialized_data == expected_data
        assert actual_data.serialization == "pickle"

    @pytest.mark.level("unit")
    def test_get_keys(self, mocker):
        mock_request = mocker.patch("runhouse.servers.http.HTTPClient.request")

        self.client.keys()
        mock_request.assert_called_with("keys", req_type="get")

        mock_request.reset_mock()

        test_env = "test_env"
        self.client.keys(env=test_env)
        mock_request.assert_called_with(f"keys/?env_name={test_env}", req_type="get")

    @pytest.mark.level("unit")
    def test_delete(self, mocker):

        mock_request = mocker.patch("runhouse.servers.http.HTTPClient.request_json")

        keys = ["key1", "key2"]

        self.client.delete(keys=keys)

        mock_request.assert_called_once_with(
            "delete_object",
            req_type="post",
            json_dict=mocker.ANY,
            err_str=f"Error deleting keys {keys}",
        )

        actual_data = DeleteObjectParams(**mock_request.call_args[1]["json_dict"])
        assert actual_data.keys == keys
