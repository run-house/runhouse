import logging
import time

import grpc

import ray.cloudpickle as pickle

import runhouse.servers.grpc.unary_pb2 as pb2

import runhouse.servers.grpc.unary_pb2_grpc as pb2_grpc

logger = logging.getLogger(__name__)


class OutputType:
    STDOUT = "stdout"
    STDERR = "stderr"
    RESULT = "result"


class UnaryClient(object):
    """
    Client for gRPC functionality
    # TODO rename ClusterClient
    """

    DEFAULT_PORT = 50052
    MAX_MESSAGE_LENGTH = 1 * 1024 * 1024 * 1024  # 1 GB
    TIMEOUT_SEC = 3

    def __init__(self, host, port=DEFAULT_PORT):
        self.host = host
        self.port = port
        self.channel = grpc.insecure_channel(
            f"{self.host}:{self.port}",
            options=[
                ("grpc.max_send_message_length", self.MAX_MESSAGE_LENGTH),
                ("grpc.max_receive_message_length", self.MAX_MESSAGE_LENGTH),
            ],
        )
        self._connectivity_state = None

        def on_state_change(state):
            self._connectivity_state = state

        self.channel.subscribe(on_state_change, False)

        # bind the client and the server
        self.stub = pb2_grpc.UnaryStub(self.channel)

        # NOTE: might be helpful for debugging purposes
        # os.environ['GRPC_TRACE'] = 'all'
        # os.environ['GRPC_VERBOSITY'] = 'DEBUG'

    def install_packages(self, to_install, env=None):
        message = pb2.Message(message=pickle.dumps((to_install, env)))
        server_res = self.stub.InstallPackages(message)
        [res, fn_exception, fn_traceback] = pickle.loads(server_res.message)
        if fn_exception is not None:
            logger.error(f"Error installing packages {to_install}: {fn_exception}")
            logger.error(f"Traceback: {fn_traceback}")
            raise fn_exception
        return res

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
        env_vars,
        run_name,
        args,
        kwargs,
    ):
        """
        Client function to call the rpc for RunModule
        """
        # Measure the time it takes to send the message
        serialized_module = pickle.dumps(
            [
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
        )
        start = time.time()
        message = pb2.Message(message=serialized_module)
        server_res = self.stub.RunModule(message)
        end = time.time()
        logging.info(f"Time to call remote function: {round(end - start, 2)} seconds")
        if server_res.result != b"":
            res = pickle.loads(server_res.result)
            return res
        if server_res.exception != b"":
            exception = pickle.loads(server_res.exception)
            logger.error(f"Error inside function {fn_type}: {exception}.")
            logger.error(f"Traceback: {server_res.traceback}")
            raise exception

    def is_connected(self):
        return self._connectivity_state in [
            grpc.ChannelConnectivity.READY,
            grpc.ChannelConnectivity.IDLE,
        ]

    def shutdown(self):
        if self.channel:
            self.channel.close()
