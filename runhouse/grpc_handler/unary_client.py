import logging
import grpc
import time

import runhouse.grpc_handler.unary_pb2_grpc as pb2_grpc
import runhouse.grpc_handler.unary_pb2 as pb2


class UnaryClient(object):
    """
    Client for gRPC functionality
    # TODO rename SendClient
    """
    DEFAULT_PORT = 50052

    def __init__(self, host, port=DEFAULT_PORT):
        self.host = host
        self.port = port
        self.channel = grpc.insecure_channel(f'{self.host}:{self.port}')
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

    def call_fn_remotely(self, message):
        """
        Client function to call the rpc for GetServerResponse
        """
        # Measure the time it takes to send the message
        start = time.time()
        message = pb2.Message(message=message)
        server_res = self.stub.GetServerResponse(message)
        end = time.time()
        logging.info(f"Time to send message: {round(end - start, 2)} seconds")
        return server_res

    def is_connected(self):
        return self._connectivity_state in [grpc.ChannelConnectivity.READY, grpc.ChannelConnectivity.IDLE]

    def shutdown(self):
        if self.channel:
            self.channel.close()
