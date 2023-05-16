import json
import logging
import time
import requests

import ray.cloudpickle as pickle

from runhouse.servers.http.http_utils import pickle_b64, b64_unpickle, OutputType, handle_response, Response
import runhouse.servers.grpc.unary_pb2 as pb2

logger = logging.getLogger(__name__)


class HTTPClient:
    """
    Client for cluster RPCs
    """

    DEFAULT_PORT = 50052
    MAX_MESSAGE_LENGTH = 1 * 1024 * 1024 * 1024  # 1 GB
    CHECK_TIMEOUT_SEC = 3

    def __init__(self, host, port=DEFAULT_PORT):
        self.host = host
        self.port = port

    def request(self, endpoint, data=None, err_str=None, timeout=None):
        response = requests.post(f"http://{self.host}:{self.port}/{endpoint}/",
                                 json={"data": data},
                                 timeout=timeout)
        output_type = response.json()["output_type"]
        return handle_response(response.json(), output_type, err_str)

    def check_server(self, cluster_config=None):
        self.request("check", data=json.dumps(cluster_config, indent=4), timeout=self.CHECK_TIMEOUT_SEC)

    def install_packages(self, to_install, env=""):
        self.request("install",
                     pickle_b64((to_install, env)),
                     err_str=f"Error installing packages {to_install}")

    def add_secrets(self, secrets):
        message = pb2.Message(message=secrets)
        server_res = self.stub.AddSecrets(message)
        return pickle.loads(server_res.message)

    def cancel_runs(self, keys, force=False, all=False):
        message = pb2.Message(message=pickle.dumps((keys, force, all)))
        res = self.stub.CancelRun(message)
        return pickle.loads(res.message)

    def list_keys(self):
        res = self.stub.ListKeys(pb2.Message())
        return pickle.loads(res.message)

    # TODO [DG]: maybe just merge cancel into this so we can get log streaming back as we cancel a job
    def get_object(self, key, stream_logs=False):
        """
        Get a value from the server
        """
        res = requests.get(f"http://{self.host}:{self.port}/get_object/",
                            json={"data": pickle_b64((key, stream_logs))})
        for responses_json in res.iter_content(chunk_size=None):
            for resp in responses_json.decode().split('{"data":')[1:]:
                resp = json.loads('{"data":'+resp)
                output_type = resp['output_type']
                result = handle_response(resp, output_type, f"Error running or getting key {key}")
                if output_type not in [OutputType.STDOUT, OutputType.STDERR]:
                    return result

    def put_object(self, key, value):
        message = pb2.Message(message=pickle.dumps((key, value)))
        resp = self.stub.PutObject(message)
        server_res = pickle.loads(resp.message)
        [res, fn_exception, fn_traceback] = server_res
        if fn_exception is not None:
            logger.error(
                f"Error putting object with key {key} on cluster: {fn_exception}."
            )
            logger.error(f"Traceback: {fn_traceback}")
            raise fn_exception
        return res

    def clear_pins(self, pins=None):
        message = pb2.Message(message=pickle.dumps(pins or []))
        self.stub.ClearPins(message)

    def run_module(
        self,
        relative_path,
        module_name,
        fn_name,
        fn_type,
        resources,
        conda_env,
        args,
        kwargs,
    ):
        """
        Client function to call the rpc for RunModule
        """
        # Measure the time it takes to send the message
        module_info = [
            relative_path,
            module_name,
            fn_name,
            fn_type,
            resources,
            conda_env,
            args,
            kwargs,
        ]
        start = time.time()
        res = self.request("run",
                           pickle_b64(module_info),
                           err_str=f"Error inside function {fn_type}")
        end = time.time()
        logging.info(f"Time to call remote function: {round(end - start, 2)} seconds")
        return res
