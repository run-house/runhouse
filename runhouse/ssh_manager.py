import os
import logging
import paramiko

logger = logging.getLogger(__name__)


class SSHManager:
    """Manage the connection via SSH to remote EC2 instance"""

    def __init__(self, hostname, username=None, path_to_pem=None, dest_dir=None):
        # read the connection details from local .env file if not provided
        self.hostname = hostname  # hostname is public IP of EC2
        self.username = username or os.getenv('EC2_USERNAME')
        self.path_to_pem = path_to_pem or os.getenv('PATH_TO_PEM')
        # Where to save the file on EC2 server
        self.dest_dir = dest_dir or os.getenv('DEST_DIR')
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
        except Exception as e:
            logger.error('Unable to connect to server', extra={'error_msg': str(e),
                                                               **self._log_meta()})

    def copy_file_to_remote_server(self, filepath: str):
        try:
            self.connect_to_server()
            ftp_client = self.client.open_sftp()
            dest_path = os.path.join(self.dest_dir, self._get_filename_from_path(filepath))
            ftp_client.put(filepath, dest_path)
            ftp_client.close()
        except Exception as e:
            logger.error('Unable to copy file to server', extra={'error_msg': str(e),
                                                                 **self._log_meta()})
