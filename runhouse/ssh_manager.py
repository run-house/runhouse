import logging
import select
import paramiko
import typer

logger = logging.getLogger(__name__)


class SSHManager:
    """Manage the connection via SSH to remote EC2 instance"""
    EC2_USERNAME = 'ec2-user'
    DEST_DIR = '/home/ec2-user/'

    def __init__(self, hostname, path_to_pem, username=None, dest_dir=None):
        # read the connection details from local .env file if not provided
        self.hostname = hostname  # hostname is public IP of EC2
        self.path_to_pem = path_to_pem
        assert self.path_to_pem is not None, "Need to specify path to pem file for creating ssh connection"
        self.username = username or self.EC2_USERNAME
        # Where to save the file on EC2 server
        self.dest_dir = dest_dir or self.DEST_DIR
        self.client = self._build_ssh_client()
        self.key = self.conn_key

    def _log_meta(self):
        return {'hostname': self.hostname, 'username': self.username,
                'path_to_pem': self.path_to_pem}

    @property
    def conn_key(self):
        return paramiko.RSAKey.from_private_key_file(filename=self.path_to_pem)

    @staticmethod
    def _build_ssh_client():
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        return client

    @staticmethod
    def _get_filename_from_path(path):
        return path.split('/')[-1]

    def connect_to_server(self):
        try:
            self.client.connect(hostname=self.hostname, username=self.username, pkey=self.key)
        except Exception:
            typer.echo('Connection error')
            raise typer.Exit(code=1)

    def create_ftp_client(self):
        self.connect_to_server()
        return self.client.open_sftp()

    def execute_command_on_remote_server(self, cmd: str, timeout=60) -> bytes:
        """Run the provided command on the server using the SSH Manager"""
        # one channel per command
        stdin, stdout, stderr = self.client.exec_command(cmd)
        # get the shared channel for stdout/stderr/stdin
        channel = stdout.channel

        # we do not need stdin.
        stdin.close()
        # indicate that we're not going to write to that channel anymore
        channel.shutdown_write()

        # read stdout/stderr in order to prevent read block hangs
        stdout_chunks = []
        stdout_chunks.append(stdout.channel.recv(len(stdout.channel.in_buffer)))
        # chunked read to prevent stalls
        while not channel.closed or channel.recv_ready() or channel.recv_stderr_ready():
            # stop if channel was closed prematurely, and there is no data in the buffers.
            got_chunk = False
            readq, _, _ = select.select([stdout.channel], [], [], timeout)
            for c in readq:
                if c.recv_ready():
                    stdout_chunks.append(stdout.channel.recv(len(c.in_buffer)))
                    got_chunk = True
                if c.recv_stderr_ready():
                    # make sure to read stderr to prevent stall
                    stderr.channel.recv_stderr(len(c.in_stderr_buffer))
                    got_chunk = True
            '''
            1) make sure that there are at least 2 cycles with no data in the input buffers in order to not exit too early (i.e. cat on a >200k file).
            2) if no data arrived in the last loop, check if we already received the exit code
            3) check if input buffers are empty
            4) exit the loop
            '''
            if not got_chunk \
                    and stdout.channel.exit_status_ready() \
                    and not stderr.channel.recv_stderr_ready() \
                    and not stdout.channel.recv_ready():
                # indicate that we're not going to read from this channel anymore
                stdout.channel.shutdown_read()
                # close the channel
                stdout.channel.close()
                break  # exit as remote side is finished and our bufferes are empty

        # close all the pseudofiles
        stdout.close()
        stderr.close()

        return b''.join(stdout_chunks)
