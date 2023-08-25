import json
import logging
import time

import requests

from runhouse.rns.resource import Resource
from runhouse.rns.utils.env import _get_env_from
from runhouse.servers.http.http_utils import handle_response, OutputType, pickle_b64

logger = logging.getLogger(__name__)


class HTTPClient:
    """
    Client for cluster RPCs
    """

    DEFAULT_PORT = 50052
    MAX_MESSAGE_LENGTH = 1 * 1024 * 1024 * 1024  # 1 GB
    CHECK_TIMEOUT_SEC = 10

    def __init__(self, host, port=DEFAULT_PORT, auth=None):
        self.host = host
        self.port = port
        self.auth = auth

    def request(
        self,
        endpoint,
        req_type="post",
        data=None,
        env=None,
        stream_logs=True,
        save=False,
        key=None,
        err_str=None,
        timeout=None,
    ):
        req_fn = (
            requests.get
            if req_type == "get"
            else requests.put
            if req_type == "put"
            else requests.delete
            if req_type == "delete"
            else requests.post
        )
        endpoint = endpoint.strip("/")
        endpoint = (endpoint + "/") if "?" not in endpoint else endpoint
        response = req_fn(
            f"http://{self.host}:{self.port}/{endpoint}",
            json={
                "data": data,
                "env": env,
                "stream_logs": stream_logs,
                "save": save,
                "key": key,
            },
            timeout=timeout,
            auth=self.auth,
        )
        if response.status_code != 200:
            raise ValueError(
                f"Error calling {endpoint} on server: {response.content.decode()}"
            )
        resp_json = response.json()
        output_type = resp_json["output_type"]
        return handle_response(resp_json, output_type, err_str)

    def check_server(self, cluster_config=None):
        self.request(
            "check",
            req_type="post",
            data=json.dumps(cluster_config, indent=4),
            timeout=self.CHECK_TIMEOUT_SEC,
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

    def call_module_method(
        self,
        module_name,
        method_name,
        env=None,
        stream_logs=True,
        save=False,
        run_name=None,
        remote=False,
        run_async=False,
        args=None,
        kwargs=None,
        system=None,
    ):
        """
        Client function to call the rpc for run_module
        """
        # Measure the time it takes to send the message
        start = time.time()
        logger.info(
            f"{'Calling' if method_name else 'Getting'} {module_name}"
            + (f".{method_name}" if method_name else "")
        )
        res = requests.post(
            f"http://{self.host}:{self.port}/{module_name}/{method_name}/",
            json={
                "data": pickle_b64([args, kwargs]),
                "env": env,
                "stream_logs": stream_logs,
                "save": save,
                "key": run_name,
                "remote": remote,
                "run_async": run_async,
            },
            stream=not run_async,
        )
        if res.status_code != 200:
            raise ValueError(
                f"Error calling {method_name} on server: {res.content.decode()}"
            )
        error_str = f"Error calling {method_name} on {module_name} on server"

        # We get back a stream of intermingled log outputs and results (maybe None, maybe error, maybe single result,
        # maybe a stream of results), so we need to separate these out.
        non_generator_result = None
        res_iter = iter(res.iter_content(chunk_size=None))
        for responses_json in res_iter:
            resp = json.loads(responses_json)
            output_type = resp["output_type"]
            result = handle_response(resp, output_type, error_str)
            if output_type in [OutputType.RESULT_STREAM, OutputType.SUCCESS_STREAM]:
                # First time we encounter a stream result, we know the rest of the results will be a stream, so return
                # a generator
                def results_generator():
                    # If this is supposed to be an empty generator, there's no first result to return
                    if not output_type == OutputType.SUCCESS_STREAM:
                        yield result
                    for responses_json_inner in res_iter:
                        resp_inner = json.loads(responses_json_inner)
                        output_type_inner = resp_inner["output_type"]
                        result_inner = handle_response(
                            resp_inner, output_type_inner, error_str
                        )
                        # if output_type == OutputType.SUCCESS_STREAM:
                        #     break
                        if output_type_inner in [
                            OutputType.RESULT_STREAM,
                            OutputType.RESULT,
                        ]:
                            yield result_inner
                    end_inner = time.time()
                    if method_name:
                        log_str = f"Time to call {module_name}.{method_name}: {round(end_inner - start, 2)} seconds"
                    else:
                        log_str = f"Time to get {module_name}: {round(end_inner - start, 2)} seconds"
                    logging.info(log_str)

                return results_generator()
            elif output_type == OutputType.CONFIG:
                # If this was a `.remote` call, we don't need to recreate the system and connection, which can be
                # slow, we can just set it explicitly.
                if (
                    system
                    and "system" in result
                    and system.rns_address == result["system"]
                ):
                    result["system"] = system
                non_generator_result = Resource.from_config(result, dryrun=True)
            elif output_type == OutputType.RESULT:
                # Finish iterating over logs before returning single result
                non_generator_result = result

        end = time.time()
        if method_name:
            log_str = f"Time to call {module_name}.{method_name}: {round(end - start, 2)} seconds"
        else:
            log_str = f"Time to get {module_name}: {round(end - start, 2)} seconds"
        logging.info(log_str)
        return non_generator_result

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

    def put_object(self, key, value, env=None):
        self.request(
            "object",
            req_type="post",
            data=pickle_b64(value),
            key=key,
            env=env,
            err_str=f"Error putting object {key}",
        )

    def put_resource(self, resource, env=None, state=None, dryrun=False):
        if env and not isinstance(env, str):
            env = _get_env_from(env)
            env = env.name
        return self.request(
            "resource",
            req_type="post",
            # TODO wire up dryrun properly
            data=pickle_b64((resource.config_for_rns, state, resource.dryrun)),
            env=env,
            err_str=f"Error putting resource {resource.name or type(resource)}",
        )

    def rename_object(self, old_key, new_key):
        self.request(
            "object",
            req_type="put",
            data=pickle_b64((old_key, new_key)),
            err_str=f"Error renaming object {old_key}",
        )

    def delete_keys(self, keys=None, env=None):
        return self.request(
            "object",
            req_type="delete",
            data=pickle_b64((keys or [])),
            env=env,
            err_str=f"Error deleting keys {keys}",
        )

    def cancel(self, key, force=False):
        # Note key can be set to "all" to cancel all runs
        return self.request(
            "cancel",
            req_type="post",
            data=pickle_b64(force),
            key=key,
            err_str=f"Error cancelling runs {key}",
        )

    def list_keys(self, env=None):
        return self.request(f"keys/?env={env}" if env else "keys", req_type="get")

    def add_secrets(self, secrets):
        failed_providers = self.request(
            "secrets",
            req_type="post",
            data=pickle_b64(secrets),
            err_str="Error sending secrets",
        )
        return failed_providers
