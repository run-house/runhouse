import json
import unittest
from unittest.mock import ANY, MagicMock, Mock, mock_open, patch

from runhouse.servers.http import HTTPClient
from runhouse.servers.http.http_utils import pickle_b64


class TestHTTPClient(unittest.TestCase):
    def setUp(self):
        self.client = HTTPClient("localhost", HTTPClient.DEFAULT_PORT)

    @patch("requests.get")
    def test_check_server(self, mock_get):
        mock_response = Mock()
        mock_response.status_code = 200
        mock_get.return_value = mock_response

        self.client.check_server()

        mock_get.assert_called_once_with(
            f"http://localhost:{HTTPClient.DEFAULT_PORT}/check",
            timeout=HTTPClient.CHECK_TIMEOUT_SEC,
            verify=False,
        )

    @patch("runhouse.servers.http.HTTPClient.request")
    @patch("pathlib.Path.mkdir")  # Mock the mkdir method
    @patch("builtins.open", new_callable=mock_open, read_data="certificate_content")
    def test_get_certificate(self, mock_file_open, mock_mkdir, mock_request):
        mock_request.return_value = b"certificate_content"
        self.client.cert_path = "/fake/path/cert.pem"

        self.client.get_certificate()

        mock_request.assert_called_once_with("cert", req_type="get", headers={})

        # Check that mkdir was called correctly
        mock_mkdir.assert_called_once()

        # Check that open was called correctly
        mock_file_open.assert_called_once_with("/fake/path/cert.pem", "wb")

        # Check that the correct content was written to the file
        mock_file_open().write.assert_called_once_with(b"certificate_content")

    @patch("pathlib.Path.exists", return_value=True)
    @patch("builtins.open", mock_open(read_data="cert_data"))
    @patch("cryptography.x509.load_pem_x509_certificate")
    def test_use_cert_verification(self, mock_load_cert, mock_exists):
        # Mock a certificate where the issuer is different from the subject
        mock_cert = MagicMock()
        mock_cert.issuer = "issuer"
        mock_cert.subject = "subject"
        mock_load_cert.return_value = mock_cert

        # Test with HTTPS enabled and a valid cert path
        client = HTTPClient(
            "localhost",
            HTTPClient.DEFAULT_PORT,
            use_https=True,
            cert_path="/valid/path",
        )
        self.assertTrue(client.verify)

        # Mock a self-signed cert where the issuer is the same as the subject
        mock_cert.issuer = "self-signed"
        mock_cert.subject = "self-signed"
        mock_load_cert.return_value = mock_cert

        # Test with HTTPS enabled and an existing cert path
        client = HTTPClient(
            "localhost",
            HTTPClient.DEFAULT_PORT,
            use_https=True,
            cert_path="/self-signed/path",
        )
        self.assertFalse(client.verify)

        # Test with HTTPS enabled and an invalid cert path
        mock_exists.return_value = False
        client = HTTPClient(
            "localhost",
            HTTPClient.DEFAULT_PORT,
            use_https=True,
            cert_path="/invalid/path",
        )
        self.assertFalse(client.verify)

    @patch("requests.post")
    def test_call_module_method(self, mock_post):
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.iter_lines.return_value = iter(
            [json.dumps({"output_type": "result", "data": "some_result"})]
        )
        mock_post.return_value = mock_response
        method_name = "install"
        module_name = "base_env"

        result = self.client.call_module_method(module_name, method_name)

        self.assertEqual(next(result), "some_result")
        mock_post.assert_called_once()

    @patch("runhouse.servers.http.HTTPClient.request")
    def test_put_object(self, mock_request):
        key = "my_list"
        value = list(range(5, 50, 2)) + ["a string"]
        expected_data = pickle_b64(value)

        self.client.put_object(key, value)

        mock_request.assert_called_once_with(
            "object",
            req_type="post",
            data=ANY,
            key=key,
            env=None,
            err_str=f"Error putting object {key}",
        )

        actual_data = mock_request.call_args[1]["data"]
        self.assertEqual(actual_data, expected_data)

    @patch("runhouse.servers.http.HTTPClient.request")
    def test_delete(self, mock_request):
        keys = ["key1", "key2"]
        expected_data = pickle_b64(keys)

        self.client.delete(keys=keys)

        mock_request.assert_called_once_with(
            "object",
            req_type="delete",
            data=ANY,
            env=None,
            err_str=f"Error deleting keys {keys}",
        )

        actual_data = mock_request.call_args[1]["data"]
        self.assertEqual(actual_data, expected_data)


# TODO [JL] test response stream parsing and error handling
@unittest.skip("Not implemented yet.")
class TestHTTPClientCallModuleMethod(unittest.TestCase):
    def setUp(self):
        # Setup code to instantiate the HTTPClient before each test
        self.client = HTTPClient(
            "localhost",
            HTTPClient.DEFAULT_PORT,
            use_https=True,
            cert_path="/valid/path",
        )

    @patch("requests.post")
    def test_call_module_method_success(self, mock_post):
        # Simulate a successful response from the server
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.iter_lines.return_value = iter(
            [json.dumps({"output_type": "result", "data": "some_result"})]
        )
        mock_post.return_value = mock_response

        result = self.client.call_module_method("module", "method")

        self.assertEqual(next(result), "some_result")
        mock_post.assert_called_once_with(
            ANY, json=ANY, stream=ANY, headers=ANY, verify=ANY
        )

    @patch("requests.post")
    def test_call_module_method_error(self, mock_post):
        mock_response = Mock()
        mock_response.status_code = 500
        mock_response.content = b"Internal Server Error"
        mock_post.return_value = mock_response

        # Call the method and expect an exception
        method_name = "install"
        module_name = "base_env"
        with self.assertRaises(ValueError) as context:
            self.client.call_module_method(module_name, method_name)

        # Assertions
        self.assertIn(f"Error calling {method_name} on server", str(context.exception))

    @patch("requests.post")
    def test_call_module_method_stream(self, mock_post):
        # Simulate a stream response from the server
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.iter_lines.return_value = iter(
            [
                json.dumps(
                    {"output_type": "result_stream", "data": "streaming_result_1"}
                ),
                json.dumps(
                    {"output_type": "result_stream", "data": "streaming_result_2"}
                ),
            ]
        )
        mock_post.return_value = mock_response

        result_generator = self.client.call_module_method("module", "method")

        results = list(result_generator)
        self.assertEqual(results, ["streaming_result_1", "streaming_result_2"])


if __name__ == "__main__":
    unittest.main()
