import json
import logging
import time

import requests

from runhouse.servers.http.http_utils import handle_response, OutputType, pickle_b64

logger = logging.getLogger(__name__)


class HTTPClient:
    """
    Client for cluster RPCs
    """

    DEFAULT_PORT = 50052
    MAX_MESSAGE_LENGTH = 1 * 1024 * 1024 * 1024  # 1 GB
    CHECK_TIMEOUT_SEC = 5

    def __init__(self, host, port=DEFAULT_PORT):
        self.host = host
        self.port = port

    def request(self, endpoint, req_type="post", data=None, err_str=None, timeout=None):
        req_fn = (
            requests.get
            if req_type == "get"
            else requests.put
            if req_type == "put"
            else requests.delete
            if req_type == "delete"
            else requests.post
        )
        response = req_fn(
            f"http://{self.host}:{self.port}/{endpoint}/",
            json={"data": data},
            timeout=timeout,
        )
        if response.status_code != 200:
            raise ValueError(
                f"Error calling {endpoint} on server: {response.content.decode()}"
            )
        output_type = response.json()["output_type"]
        return handle_response(response.json(), output_type, err_str)

    def check_server(self, cluster_config=None):
        self.request(
            "check",
            req_type="post",
            data=json.dumps(cluster_config, indent=4),
            timeout=self.CHECK_TIMEOUT_SEC,
        )

    def install(self, to_install, env=""):
        self.request(
            "env",
            req_type="post",
            data=pickle_b64((to_install, env)),
            err_str=f"Error installing packages {to_install}",
        )

    def run_module(
        self,
        relative_path,
        module_name,
        fn_name,
        fn_type,
        resources,
        conda_env,
        env_vars,
        run_name,
        args,
        kwargs,
    ):
        """
        Client function to call the rpc for run_module
        """
        # Measure the time it takes to send the message
        module_info = [
            relative_path,
            module_name,
            fn_name,
            fn_type,
            resources,
            conda_env,
            env_vars,
            run_name,
            args,
            kwargs,
        ]
        start = time.time()
        res = self.request(
            "run",
            req_type="post",
            data=pickle_b64(module_info),
            err_str=f"Error inside function {fn_type}",
        )
        end = time.time()
        if fn_type not in ["remote", "get_or_run"]:
            # Printing call time for async runs is not useful
            logging.info(
                f"Time to call remote function: {round(end - start, 2)} seconds"
            )
        return res

    # TODO [DG]: maybe just merge cancel into this so we can get log streaming back as we cancel a job (ditto others)
    def get_object(self, key, stream_logs=False):
        """
        Get a value from the server
        """
        res = requests.get(
            f"http://{self.host}:{self.port}/object/",
            json={"data": pickle_b64((key, stream_logs))},
            stream=True,
        )
        if res.status_code != 200:
            raise ValueError(
                f"Error getting key {key} from server: {res.content.decode()}"
            )
        for responses_json in res.iter_content(chunk_size=None):
            for resp in responses_json.decode().split('{"data":')[1:]:
                resp = json.loads('{"data":' + resp)
                output_type = resp["output_type"]
                result = handle_response(
                    resp, output_type, f"Error running or getting key {key}"
                )
                if output_type not in [OutputType.STDOUT, OutputType.STDERR]:
                    return result

    def put_object(self, key, value):
        self.request(
            "object",
            req_type="put",
            data=pickle_b64((key, value)),
            err_str=f"Error putting object {key}",
        )

    def get_run_object(self, run_name, folder_path):
        run_obj = self.request(
            "run_object",
            req_type="get",
            data=pickle_b64((run_name, folder_path)),
            err_str=f"Error getting Run with name {run_name}",
        )

        return run_obj

    def clear_pins(self, pins=None):
        return self.request(
            "object",
            req_type="delete",
            data=pickle_b64((pins or [])),
            err_str=f"Error clearing pins {pins}",
        )

    def cancel_runs(self, keys, force=False):
        # Note keys can be set to "all" to cancel all runs
        return self.request(
            "cancel",
            req_type="post",
            data=pickle_b64((keys, force)),
            err_str=f"Error cancelling runs {keys}",
        )

    def list_keys(self):
        return self.request("keys", req_type="get")

    def add_secrets(self, secrets):
        failed_providers = self.request(
            "secrets",
            req_type="post",
            data=pickle_b64(secrets),
            err_str="Error sending secrets",
        )
        return failed_providers
