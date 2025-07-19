# Runhouse exceptions


class InsufficientDiskError(Exception):
    """Raised when a process on the cluster fails due to lack of disk space.

    Args:
        command: The command / process that was run.
        error_msg: The error message to print.
    """

    def __init__(
        self,
        error_msg: str = None,
        command: str = None,
    ) -> None:
        self.command = command
        self.error_msg = error_msg
        self.default_error_msg = "Cluster is out of disk space"
        msg = (
            self.error_msg
            if self.error_msg
            else f"Command {command} failed"
            if self.command
            else self.default_error_msg
        )
        msg = f"{msg}. To resolve it, teardown the cluster and re-launch it with larger disk size."
        super().__init__(msg)


class SerializationError(Exception):
    """Raised when we have serialization error.

    Args:
        error_msg: The error message to print.
    """

    def __init__(
        self,
        error_msg: str = None,
    ) -> None:
        self.error_msg = error_msg
        self.default_error_msg = "Got a serialization error."
        msg = self.error_msg if self.error_msg else self.default_error_msg
        msg = f"{msg}. Make sure that the remote and local versions of python and all installed packages are as expected.\n Please Check logs for more information."
        super().__init__(msg)


class RemoteException(Exception):
    """Raised when an exception is raised remotely on the cluster, but we can't raise it properly on the http client.
    Serves as a wrapper to the remote exception.

    Args:
        error_msg: The error msg printed remotely.
        exception_type: type of the remote exception
        traceback: remote exception's traceback
        args: args passed to the remote exception when raised on the cluster
    """

    def __init__(
        self,
        error_msg: str = None,
        exception_type: str = None,
        traceback: str = None,
        args: tuple = (),
    ) -> None:
        self.error_msg = (
            error_msg
            if error_msg
            else "An exception was raised remotely. See logs for more info."
        )
        self.exception_type = exception_type
        self.remote_args = args
        self.remote_traceback = traceback
        full_msg = (
            f"{self.exception_type}: {self.error_msg} was raised remotely."
            if self.exception_type
            else self.error_msg
        )
        super().__init__(full_msg)
