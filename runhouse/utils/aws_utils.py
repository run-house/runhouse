"""
- Pushing an image to AWS's Elastic Container Registry (ECR)
- Pushing a file to S3 bucket
Assumes that the AWS account info has been set up and Docker is running on the host machine
"""

import base64
import json
import os
import boto3
import typer
from botocore.exceptions import ClientError
from runhouse.utils.utils import ERROR_FLAG


def aws_credentials():
    # get AWS credentials
    aws_credentials = read_aws_credentials()
    access_key_id = aws_credentials['access_key_id']
    secret_access_key = aws_credentials['secret_access_key']
    aws_region = aws_credentials['region']
    return access_key_id, secret_access_key, aws_region


def aws_client_creator(client_type):
    access_key_id, secret_access_key, aws_region = aws_credentials()
    try:
        boto3.client(client_type, aws_access_key_id=access_key_id, aws_secret_access_key=secret_access_key,
                     region_name=aws_region)
    except:
        typer.echo(f'{ERROR_FLAG} Unable to create boto client for {client_type}')
        raise typer.Exit(code=1)


def build_s3_client():
    return aws_client_creator('s3')


def build_ecr_client():
    return aws_client_creator('ecr')


def file_exists_in_s3(s3_client, file_name):
    try:
        resp = s3_client.head_object(Bucket=os.getenv('BUCKET_NAME'), Key=file_name)
        return resp
    except ClientError:
        # object doesn't already exist - let's upload it
        pass


def upload_file_to_s3(file_obj, file_name):
    s3_client = build_s3_client()
    if file_exists_in_s3(s3_client, file_name):
        # No reason to re-upload if the file already exists
        return

    try:
        with open(file_obj, 'rb') as tar:
            s3_client.upload_fileobj(tar, os.getenv('BUCKET_NAME'), file_name)
    except:
        typer.echo(f'{ERROR_FLAG} Failed to upload to s3')
        raise typer.Exit(code=1)


def push_image_to_ecr(docker_client, image, tag_name):
    """push docker image to AWS and update ECS service."""
    try:
        ecr_client: boto3.client = build_ecr_client()
        ecr_credentials = (ecr_client.get_authorization_token()['authorizationData'][0])
        ecr_username = 'AWS'
        ecr_password = (base64.b64decode(ecr_credentials['authorizationToken']).replace(b'AWS:', b'').decode('utf-8'))

        ecr_url = ecr_credentials['proxyEndpoint']
        # get Docker to login/authenticate with ECR
        docker_client.login(username=ecr_username, password=ecr_password, registry=ecr_url)

        # tag image for AWS ECR
        ecr_repo_name = f"{ecr_url.replace('https://', '')}/{os.getenv('ECR_REPOSITORY_NAME')}:latest"
        image.tag(ecr_repo_name, tag=tag_name)

        # push image to AWS ECR
        push_log = docker_client.images.push(ecr_repo_name, tag=tag_name)

    except Exception:
        typer.echo('Unable to save image')
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
