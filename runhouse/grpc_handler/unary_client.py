import logging
import sys
import time

import grpc

import ray.cloudpickle as pickle

import runhouse.grpc_handler.unary_pb2 as pb2

import runhouse.grpc_handler.unary_pb2_grpc as pb2_grpc

logger = logging.getLogger(__name__)


class OutputType:
    STDOUT = "stdout"
    STDERR = "stderr"
    RESULT = "result"


class UnaryClient(object):
    """
    Client for gRPC functionality
    # TODO rename SendClient
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

    def install_packages(self, message):
        message = pb2.Message(message=message)
        server_res = self.stub.InstallPackages(message)
        return server_res

    def cancel_runs(self, keys, force=False):
        message = pb2.Message(message=pickle.dumps((keys, force)))
        res = self.stub.CancelRun(message)
        return pickle.loads(res.message)

    # TODO [DG]: maybe just merge cancel into this so we can get log streaming back as we cancel a job
    def get_object(self, key, stream_logs=False):
        """
        Get a value from the server
        """
        message = pb2.Message(message=pickle.dumps((key, stream_logs)))
        for resp in self.stub.GetObject(message):
            output_type = resp.output_type
            server_res = pickle.loads(resp.message)
            if output_type == OutputType.STDOUT:
                for line in server_res:
                    print(line, end="", flush=True)
            elif output_type == OutputType.STDERR:
                for line in server_res:
                    print(line, file=sys.stderr)
            else:
                [res, fn_exception, fn_traceback] = server_res
                if fn_exception is not None:
                    logger.error(
                        f"Error running or getting run_key {key}: {fn_exception}."
                    )
                    logger.error(f"Traceback: {fn_traceback}")
                    raise fn_exception
                return res

    def clear_pins(self, pins=None):
        message = pb2.Message(message=pickle.dumps(pins or []))
        self.stub.ClearPins(message)

    def run_module(self, relative_path, module_name, fn_name, fn_type, args, kwargs):
        """
        Client function to call the rpc for RunModule
        """
        # Measure the time it takes to send the message
        serialized_module = pickle.dumps(
            [relative_path, module_name, fn_name, fn_type, args, kwargs]
        )
        start = time.time()
        message = pb2.Message(message=serialized_module)
        server_res = self.stub.RunModule(message)
        end = time.time()
        logging.info(f"Time to send message: {round(end - start, 2)} seconds")
        [res, fn_exception, fn_traceback] = pickle.loads(server_res.message)
        if fn_exception is not None:
            logger.error(f"Error inside send {fn_type}: {fn_exception}.")
            logger.error(f"Traceback: {fn_traceback}")
            raise fn_exception
        return res

    def is_connected(self):
        return self._connectivity_state in [
            grpc.ChannelConnectivity.READY,
            grpc.ChannelConnectivity.IDLE,
        ]

    def shutdown(self):
        if self.channel:
            self.channel.close()
