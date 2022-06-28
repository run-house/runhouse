import warnings

# Suppress warnings when running the tests
from cryptography.utils import CryptographyDeprecationWarning

warnings.filterwarnings("ignore", category=CryptographyDeprecationWarning)

import base64
import os
import json
import random
from runhouse.utils.deploy_to_aws import build_ecr_client
from runhouse.ssh_manager import SSHManager
from runhouse.main import get_hostname_from_hardware
from runhouse.utils.validation import valid_filepath

from dotenv import load_dotenv

load_dotenv()


def create_ssh_manager_for_server(hostname):
    ssh_manager = SSHManager(hostname=hostname, path_to_pem=os.getenv('PATH_TO_PEM'))
    ssh_manager.connect_to_server()
    return ssh_manager


def select_random_server():
    hardware_to_hostname = bring_all_server_data()
    random_hardware = random.choice(list(hardware_to_hostname))
    random_hostname = get_hostname_from_hardware(hardware=random_hardware)
    return random_hostname


def bring_all_server_data():
    return json.loads(os.getenv('HARDWARE_TO_HOSTNAME'))


# --------------------------------------

def test_valid_pem_file():
    path_to_pem = os.getenv('PATH_TO_PEM')
    if path_to_pem is None or not valid_filepath(path_to_pem):
        raise Exception('Invalid path to pem file')


def test_ssh_connection_to_all_instances():
    hardware_to_hostname = bring_all_server_data()
    for hardware, hostname in hardware_to_hostname.items():
        try:
            create_ssh_manager_for_server(hostname)
        except:
            raise Exception(f'Failed to connect to {hostname} with hardware {hardware}')


def test_connection_to_ecr():
    try:
        build_ecr_client()
    except:
        raise Exception(f'Unable to connect to ECR and bring token')


def test_docker_connection_on_server():
    hostname = select_random_server()
    try:
        ssh_manager = create_ssh_manager_for_server(hostname)

        command = f'sudo systemctl status docker'
        stdout = ssh_manager.execute_command_on_remote_server(command)
        # TODO find an easier way to check this?
        if '(running)' not in str(stdout).split('active')[1].strip():
            raise Exception(f'Docker daemon not running on {hostname}')
    except:
        raise Exception(f'Unable to run docker command on server {hostname}')


def test_docker_login_credentials():
    hostname = select_random_server()
    try:
        ecr_client = build_ecr_client()
        token = ecr_client.get_authorization_token()
        username, password = base64.b64decode(token['authorizationData'][0]['authorizationToken']).decode().split(':')
        registry = token['authorizationData'][0]['proxyEndpoint']

        # login in via exec command
        command = f'docker login -u {username} -p {password} {registry}'

        ssh_manager = create_ssh_manager_for_server(hostname)
        stdout = ssh_manager.execute_command_on_remote_server(command)
        if stdout.decode('utf-8').strip() != 'Login Succeeded':
            raise Exception(f'Docker login failed for {hostname}')
    except:
        raise Exception(f'Failed to login to {hostname} with ecr credentials')
