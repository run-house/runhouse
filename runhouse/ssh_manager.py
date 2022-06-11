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

    def create_ftp_client(self):
        self.connect_to_server()
        return self.client.open_sftp()

    def copy_file_to_remote_server(self, ftp_client, filepath: str):
        try:
            dest_path = os.path.join(self.dest_dir, self._get_filename_from_path(filepath))
            ftp_client.put(filepath, dest_path)
            ftp_client.close()
        except Exception as e:
            logger.error('Unable to copy file to server', extra={'error_msg': str(e),
                                                                 **self._log_meta()})

    def put_dir(self, ftp_client, source_dir, target_dir=None):
        ''' Uploads the contents of the source directory to the target path. The
            target directory needs to exists. All subdirectories in source are
            created under target.
        '''
        target_dir = target_dir or self.dest_dir
        for item in os.listdir(source_dir):
            if os.path.isfile(os.path.join(source_dir, item)):
                ftp_client.put(os.path.join(source_dir, item), '%s/%s' % (target_dir, item))
            else:
                self.create_dir(ftp_client, '%s/%s' % (target_dir, item), ignore_existing=True)
                self.put_dir(os.path.join(source_dir, item), '%s/%s' % (target_dir, item))

    @staticmethod
    def create_dir(ftp_client, path, mode=511, ignore_existing=False):
        ''' Augments mkdir by adding an option to not fail if the folder exists  '''
        try:
            ftp_client.mkdir(path, mode)
        except IOError:
            if ignore_existing:
                pass
            else:
                raise