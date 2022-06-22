import logging
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

    def execute_command_on_remote_server(self, cmd: str, read_lines=False):
        """Run the provided command on the server using the SSH Manager"""
        stdin, stdout, stderr = self.client.exec_command(cmd)
        if read_lines:
            stdin.flush()
            stdout = stdout.read().splitlines()
        else:
            stdout = stdout.readlines()
        stdin.close()
        return stdout
