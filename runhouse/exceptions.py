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
