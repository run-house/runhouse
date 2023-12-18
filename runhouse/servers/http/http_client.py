import json
import logging
import time
import warnings
from pathlib import Path
from typing import Any, Dict, Union

import requests

from runhouse.globals import rns_client

from runhouse.resources.envs.utils import _get_env_from

from runhouse.resources.resource import Resource
from runhouse.servers.http.http_utils import handle_response, OutputType, pickle_b64

logger = logging.getLogger(__name__)


class HTTPClient:
    """
    Client for cluster RPCs
    """

    DEFAULT_PORT = 32300
    MAX_MESSAGE_LENGTH = 1 * 1024 * 1024 * 1024  # 1 GB
    CHECK_TIMEOUT_SEC = 10

    def __init__(
        self,
        host: str,
        port: int,
        auth=None,
        cert_path=None,
        use_https=False,
        system=None,
    ):
        self.host = host
        self.port = port
        self.auth = auth
        self.cert_path = cert_path
        self.use_https = use_https
        self.verify = self._use_cert_verification()
        self.system = system

    def _use_cert_verification(self):
        if not self.use_https:
            return False

        from cryptography import x509
        from cryptography.hazmat.backends import default_backend

        cert_path = Path(self.cert_path)
        if not cert_path.exists():
            return False

        # Check whether the cert is self-signed, if so we cannot use verification
        with open(cert_path, "rb") as cert_file:
            cert = x509.load_pem_x509_certificate(cert_file.read(), default_backend())

        if cert.issuer == cert.subject:
            warnings.warn(
                f"Cert in use ({cert_path}) is self-signed, cannot verify in requests to server."
            )
            return False

        return True

    @staticmethod
    def from_endpoint(endpoint: str, auth=None, cert_path=None):
        protocol, uri = endpoint.split("://")
        host, port_and_route = uri.split(":", 1)
        port, _ = port_and_route.split("/", 1)
        use_https = protocol == "https"
        client = HTTPClient(host, int(port), auth, cert_path, use_https=False)
        client.use_https = use_https
        return client

    def _formatted_url(self, endpoint: str):
        prefix = "https" if self.use_https else "http"
        return f"{prefix}://{self.host}:{self.port}/{endpoint}"

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
        headers: Union[Dict, None] = None,
    ):
        # Support use case where we explicitly do not want to provide headers (e.g. requesting a cert)
        headers = rns_client.request_headers if headers != {} else headers
        req_fn = (
            requests.get
            if req_type == "get"
            else requests.put
            if req_type == "put"
            else requests.delete
            if req_type == "delete"
            else requests.post
        )
        # Note: For localhost (e.g. docker) do not add trailing slash (will lead to connection errors)
        endpoint = endpoint.strip("/")
        if (
            self.host not in ["localhost", "127.0.0.1", "0.0.0.0"]
            and "?" not in endpoint
        ):
            endpoint += "/"

        response = req_fn(
            self._formatted_url(endpoint),
            json={
                "data": data,
                "env": env,
                "stream_logs": stream_logs,
                "save": save,
                "key": key,
            },
            timeout=timeout,
            auth=self.auth,
            headers=headers,
            verify=self.verify,
        )
        if response.status_code != 200:
            raise ValueError(
                f"Error calling {endpoint} on server: {response.content.decode()}"
            )
        resp_json = response.json()
        output_type = resp_json["output_type"]
        return handle_response(resp_json, output_type, err_str)

    def check_server(self):
        resp = requests.get(
            self._formatted_url("check"),
            timeout=self.CHECK_TIMEOUT_SEC,
            verify=self.verify,
        )

        if resp.status_code != 200:
            raise ValueError(
                f"Error checking server: {resp.content.decode()}. Is the server running?"
            )

        rh_version = resp.json().get("rh_version", None)
        import runhouse

        if rh_version and not runhouse.__version__ == rh_version:
            logger.warning(
                f"Server was started with Runhouse version ({rh_version}), "
                f"but local Runhouse version is ({runhouse.__version__})"
            )

    def get_certificate(self):
        cert: bytes = self.request(
            "cert",
            req_type="get",
            headers={},
        )
        # Create parent directory to store the cert
        Path(self.cert_path).parent.mkdir(parents=True, exist_ok=True)
        with open(self.cert_path, "wb") as file:
            file.write(cert)

    def call(
        self,
        module_name,
        method_name,
        *args,
        stream_logs=True,
        run_name=None,
        remote=False,
        run_async=False,
        save=False,
        **kwargs,
    ):
        """wrapper to temporarily support cluster's call signature"""
        return self.call_module_method(
            module_name,
            method_name,
            stream_logs=stream_logs,
            run_name=run_name,
            remote=remote,
            run_async=run_async,
            save=save,
            args=args,
            kwargs=kwargs,
            system=self.system,
        )

    def call_module_method(  # TODO rename call_module_method to call
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
        Client function to call the rpc for call_module_method
        """
        # Measure the time it takes to send the message
        start = time.time()
        logger.info(
            f"{'Calling' if method_name else 'Getting'} {module_name}"
            + (f".{method_name}" if method_name else "")
        )
        res = requests.post(
            self._formatted_url(f"{module_name}/{method_name}"),
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
            headers=rns_client.request_headers,
            verify=self.verify,
        )
        if res.status_code != 200:
            raise ValueError(
                f"Error calling {method_name} on server: {res.content.decode()}"
            )
        error_str = f"Error calling {method_name} on {module_name} on server"

        # We get back a stream of intermingled log outputs and results (maybe None, maybe error, maybe single result,
        # maybe a stream of results), so we need to separate these out.
        non_generator_result = None
        res_iter = res.iter_lines(chunk_size=None)
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
            env = env.name or env.env_name
        config = resource.config_for_rns
        return self.request(
            "resource",
            req_type="post",
            # TODO wire up dryrun properly
            data=pickle_b64((config, state, resource.dryrun)),
            env=env,
            err_str=f"Error putting resource {resource.name or type(resource)}",
        )

    def get(
        self, key: str, default: Any = None, remote=False, stream_logs: bool = False
    ):
        """Provides compatibility with cluster's get."""
        try:
            res = self.call_module_method(
                key,
                None,
                remote=remote,
                stream_logs=stream_logs,
                system=self,
            )
        except KeyError as e:
            if default == KeyError:
                raise e
            return default
        return res

    def rename(self, old_key: str, new_key: str):
        """Provides compatibility with cluster's rename."""
        return self.rename_object(old_key, new_key)

    def rename_object(self, old_key, new_key):
        self.request(
            "object",
            req_type="put",
            data=pickle_b64((old_key, new_key)),
            key=old_key,
            err_str=f"Error renaming object {old_key}",
        )

    def delete(self, keys=None, env=None):
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

    def keys(self, env=None):
        if env is not None and not isinstance(env, str):
            env = _get_env_from(env)
            env = env.name
        return self.request(f"keys/?env={env}" if env else "keys", req_type="get")
