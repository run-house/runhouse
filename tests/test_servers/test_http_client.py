import inspect
import json

import pytest

import runhouse as rh
from runhouse.constants import DEFAULT_SERVER_PORT

from runhouse.globals import rns_client

from runhouse.servers.http import HTTPClient
from runhouse.servers.http.http_utils import (
    DeleteObjectParams,
    pickle_b64,
    PutObjectParams,
)


@pytest.mark.servertest
class TestHTTPClient:
    @pytest.fixture(autouse=True)
    def init_fixtures(self):
        args = dict(name="local-cluster", host="localhost", server_host="0.0.0.0")
        self.local_cluster = rh.cluster(**args)
        self.client = HTTPClient("localhost", DEFAULT_SERVER_PORT)

    @pytest.mark.level("unit")
    def test_check_server(self, mocker):
        mock_response = mocker.Mock()
        mock_response.status_code = 200
        rh_version_resp = {"rh_version": rh.__version__}
        mock_response.json.return_value = rh_version_resp
        mocked_get = mocker.patch("requests.Session.get", return_value=mock_response)

        self.client.check_server()

        mocked_get.assert_called_once_with(
            f"http://localhost:{DEFAULT_SERVER_PORT}/check",
            timeout=HTTPClient.CHECK_TIMEOUT_SEC,
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
    def test_use_cert_verification(self, mocker):
        # Mock a certificate where the issuer is different from the subject
        mock_cert = mocker.MagicMock()
        mock_cert.issuer = "issuer"
        mock_cert.subject = "subject"

        mock_exists = mocker.patch("pathlib.Path.exists", return_value=True)
        mock_load_cert = mocker.patch(
            "builtins.open", mocker.mock_open(read_data="cert_data")
        )
        mocker.patch(
            "cryptography.x509.load_pem_x509_certificate", return_value=mock_cert
        )

        mock_load_cert.return_value = mock_cert

        # Test with HTTPS enabled and a valid cert path
        client = HTTPClient(
            "localhost",
            DEFAULT_SERVER_PORT,
            use_https=True,
            cert_path="/valid/path",
        )
        assert client.verify

        # Mock a self-signed cert where the issuer is the same as the subject
        mock_cert.issuer = "self-signed"
        mock_cert.subject = "self-signed"
        mock_load_cert.return_value = mock_cert

        # Test with HTTPS enabled and an existing cert path
        client = HTTPClient(
            "localhost",
            DEFAULT_SERVER_PORT,
            use_https=True,
            cert_path="/self-signed/path",
        )
        assert not client.verify

        # Test with HTTPS enabled and an invalid cert path
        mock_exists.return_value = False
        client = HTTPClient(
            "localhost",
            DEFAULT_SERVER_PORT,
            use_https=True,
            cert_path="/invalid/path",
        )
        assert not client.verify

    @pytest.mark.level("unit")
    def test_call_module_method(self, mocker):

        response_sequence = [
            json.dumps({"output_type": "log", "data": "Log message"}),
            json.dumps(
                {"output_type": "result_stream", "data": pickle_b64("stream_result_1")}
            ),
            json.dumps(
                {"output_type": "result_stream", "data": pickle_b64("stream_result_2")}
            ),
            json.dumps({"output_type": "result", "data": pickle_b64("final_result")}),
        ]

        # Mock the response to iter_lines to return our simulated server response
        mock_response = mocker.Mock()
        mock_response.status_code = 200
        mock_response.iter_lines.return_value = iter(response_sequence)
        mock_post = mocker.patch("requests.Session.post", return_value=mock_response)

        # Call the method under test
        method_name = "install"
        module_name = "base_env"
        result_generator = self.client.call_module_method(module_name, method_name)

        # Iterate through the generator and collect results
        results = []
        for result in result_generator:
            results.append(result)

        expected_results = ["stream_result_1", "stream_result_2", "final_result"]
        assert results == expected_results

        # Assert that the post request was called correctly
        expected_url = self.client._formatted_url(f"{module_name}/{method_name}")
        expected_json_data = {
            "data": pickle_b64([None, None]),
            "env": None,
            "stream_logs": True,
            "save": False,
            "key": None,
            "remote": False,
            "run_async": False,
        }
        expected_headers = rns_client.request_headers()

        mock_post.assert_called_once_with(
            expected_url,
            json=expected_json_data,
            stream=True,
            headers=expected_headers,
        )

    @pytest.mark.level("unit")
    def test_call_module_method_with_args_kwargs(self, mocker):
        mock_response = mocker.MagicMock()
        mock_response.status_code = 200
        # Set up iter_lines to return an iterator
        mock_response.iter_lines.return_value = iter(
            [
                json.dumps({"output_type": "log", "data": "Log message"}),
            ]
        )
        mock_post = mocker.patch("requests.Session.post", return_value=mock_response)

        args = [1, 2]
        kwargs = {"a": 3, "b": 4}
        module_name = "module"
        method_name = "install"

        self.client.call_module_method(
            module_name, method_name, args=args, kwargs=kwargs
        )

        # Assert that the post request was called with the correct data
        expected_json_data = {
            "data": pickle_b64([args, kwargs]),
            "env": None,
            "stream_logs": True,
            "save": False,
            "key": None,
            "remote": False,
            "run_async": False,
        }
        expected_url = f"http://localhost:32300/{module_name}/{method_name}"
        expected_headers = rns_client.request_headers()

        mock_post.assert_called_with(
            expected_url,
            json=expected_json_data,
            stream=True,
            headers=expected_headers,
        )

    @pytest.mark.level("unit")
    def test_call_module_method_error_handling(self, mocker):
        mock_response = mocker.Mock()
        mock_response.status_code = 500
        mock_response.content = b"Internal Server Error"
        mocker.patch("requests.Session.post", return_value=mock_response)

        with pytest.raises(ValueError):
            self.client.call_module_method("module", "method")

    @pytest.mark.level("unit")
    def test_call_module_method_stream_logs(self, mocker):
        # Setup the mock response with a log in the stream
        response_sequence = [
            json.dumps(
                {"output_type": "result_stream", "data": pickle_b64("Log message")}
            ),
        ]
        mock_response = mocker.Mock()
        mock_response.status_code = 200
        mock_response.iter_lines.return_value = iter(response_sequence)
        mocker.patch("requests.Session.post", return_value=mock_response)

        # Call the method under test
        res = self.client.call_module_method("base_env", "install")
        assert inspect.isgenerator(res)
        assert next(res) == "Log message"

    @pytest.mark.level("unit")
    def test_call_module_method_config(self, mocker):
        test_data = self.local_cluster.config_for_rns
        mock_response = mocker.Mock()
        mock_response.status_code = 200
        mock_response.iter_lines.return_value = iter(
            [
                json.dumps({"output_type": "config", "data": test_data}),
            ]
        )
        mocker.patch("requests.Session.post", return_value=mock_response)

        cluster = self.client.call_module_method("base_env", "install")
        assert cluster.config_for_rns == test_data

    @pytest.mark.level("unit")
    def test_call_module_method_not_found_error(self, mocker):
        mock_response = mocker.Mock()
        mock_response.status_code = 200
        missing_key = "missing_key"
        mock_response.iter_lines.return_value = iter(
            [
                json.dumps({"output_type": "not_found", "data": missing_key}),
            ]
        )
        mocker.patch("requests.Session.post", return_value=mock_response)

        with pytest.raises(KeyError) as context:
            next(self.client.call_module_method("module", "method"))

        assert f"key {missing_key} not found" in str(context)

    @pytest.mark.level("unit")
    def test_put_object(self, mocker):

        mock_request = mocker.patch("runhouse.servers.http.HTTPClient.request_json")

        key = "my_list"
        value = list(range(5, 50, 2)) + ["a string"]
        expected_data = pickle_b64(value)

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
