import subprocess
import time
import unittest

import pytest
import requests

from runhouse.servers.http import http_server

TEST_DEFAULT_PORT = http_server.HTTPServer.DEFAULT_PORT


def wait_for_server_to_start(port, timeout=30):
    start_time = time.time()
    while True:
        try:
            response = requests.get(f"http://localhost:{port}/docs")
            if response.status_code == 200:
                print("Server is up and running!")
                return True
        except requests.ConnectionError:
            pass

        if time.time() - start_time > timeout:
            print("Timed out waiting for server to start.")
            return False

        time.sleep(1)


def run_server(process, enable_local_span_collection=None):
    if enable_local_span_collection:
        print("\nEnabling local span collection")
        process = subprocess.Popen(
            [
                "python",
                "-m",
                "runhouse.servers.http.http_server",
                "--enable_local_span_collection=True",
                "--port",
                f"{TEST_DEFAULT_PORT}",
            ],
        )
    else:
        print("Not enabling local span collection")
        process = subprocess.Popen(
            ["python", "-m", "runhouse.servers.http.http_server"]
        )

    wait_for_server_to_start(TEST_DEFAULT_PORT)
    return process


def terminate_server(process):
    # Terminate the server
    process.terminate()
    time.sleep(1)  # Wait for server to terminate


def kill_port(port):
    try:
        subprocess.run(
            "lsof -i :%d | awk 'NR>1 {print $2}' | xargs kill -9" % port,
            shell=True,
            check=True,
        )
    except subprocess.CalledProcessError as e:
        print(f"Command failed with error: {e}")


class TestHTTPServer(unittest.TestCase):
    process: any

    @classmethod
    def setUpClass(cls):
        kill_port(TEST_DEFAULT_PORT)

    @classmethod
    def tearDownClass(cls):
        kill_port(TEST_DEFAULT_PORT)

    def setUp(self):
        self.process = None

    def tearDown(self):
        terminate_server(self.process)

    @pytest.mark.httpservertest
    def test_spans_endpoint(self):
        # Start the server enabling local span collection
        self.process = run_server(self.process, enable_local_span_collection=True)

        # Make a GET request to the /spans endpoint
        response = requests.get(f"http://127.0.0.1:{TEST_DEFAULT_PORT}/spans")
        print("Response: ", response)

        # Check the status code
        self.assertEqual(response.status_code, 200)

        # JSON parse the response
        parsed_response = response.json()
        print("Parsed response: ", parsed_response)

        # Assert that the key "spans" exists in the parsed response
        self.assertIn("spans", parsed_response)


if __name__ == "__main__":
    unittest.main()
