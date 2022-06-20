"""
A simple script that demonstrates how the docker and AWS Python clients
can be used to automate the process of: building a Docker image, as
defined by the Dockerfile in the project's root directory; pushing the
image to AWS's Elastic Container Registry (ECR)

For now, it is assumed that the AWS infrastructure is already in
existence and that Docker is running on the host machine.
"""

import base64
import json
import os

import boto3
import typer

LOCAL_REPOSITORY = 'runhouse-demo:latest'


def build_ecr_client():
    # get AWS credentials
    aws_credentials = read_aws_credentials()
    access_key_id = aws_credentials['access_key_id']
    secret_access_key = aws_credentials['secret_access_key']
    aws_region = aws_credentials['region']

    # get AWS ECR login token
    ecr_client = boto3.client('ecr', aws_access_key_id=access_key_id, aws_secret_access_key=secret_access_key,
                              region_name=aws_region)

    return ecr_client


def push_image_to_ecr(docker_client, image, tag_name):
    """push docker image to AWS and update ECS service."""
    try:
        ecr_client = build_ecr_client()
        ecr_credentials = (ecr_client.get_authorization_token()['authorizationData'][0])
        ecr_username = 'AWS'
        ecr_password = (base64.b64decode(ecr_credentials['authorizationToken']).replace(b'AWS:', b'').decode('utf-8'))

        ecr_url = ecr_credentials['proxyEndpoint']

        # get Docker to login/authenticate with ECR
        docker_client.login(username=ecr_username, password=ecr_password, registry=ecr_url)

        # tag image for AWS ECR
        ecr_repo_name = '{}/{}'.format(ecr_url.replace('https://', ''), LOCAL_REPOSITORY)
        print("ecr_repo_name", ecr_repo_name)
        image.tag(ecr_repo_name, tag=tag_name)

        # push image to AWS ECR
        push_log = docker_client.images.push(ecr_repo_name, tag=tag_name)

    except Exception as e:
        typer.echo('Unable to save image', e)
        raise typer.Exit(code=1)


def read_aws_credentials(filename='.aws_credentials.json'):
    """Read AWS credentials from file.

    :param filename: Credentials filename, defaults to '.aws_credentials.json'
    :param filename: str, optional
    :return: Dictionary of AWS credentials.
    :rtype: Dict[str, str]
    """

    try:
        with open(filename) as json_data:
            credentials = json.load(json_data)

        for variable in ('access_key_id', 'secret_access_key', 'region'):
            if variable not in credentials.keys():
                msg = '"{}" cannot be found in {}'.format(variable, filename)
                raise KeyError(msg)

    except FileNotFoundError:
        try:
            credentials = {
                'access_key_id': os.environ['AWS_ACCESS_KEY_ID'],
                'secret_access_key': os.environ['AWS_SECRET_ACCESS_KEY'],
                'region': os.environ['AWS_REGION']
            }
        except KeyError:
            msg = 'no AWS credentials found in file or environment variables'
            raise RuntimeError(msg)

    return credentials
