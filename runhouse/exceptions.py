# Runhouse exceptions


class InsufficientDisk(Exception):
    """Raised when a process on the cluster fails due to lack of disk space.

    Args:
        command: The command / process that was run.
        error_message: The error message to print.
        detailed_reason: The stderr of the command.
    """

    def __init__(
        self,
        error_msg: str = None,
        command: str = None,
    ) -> None:
        self.command = command
        self.error_msg = error_msg
        msg = (
            self.error_msg
            if self.error_msg
            else f"Command {command} failed"
            if command
            else None
        )
        super().__init__(msg)
