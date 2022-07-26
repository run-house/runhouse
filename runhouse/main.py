"""This module provides the runhouse CLI."""
import warnings

# Suppress warnings when running commands
from cryptography.utils import CryptographyDeprecationWarning
warnings.filterwarnings("ignore", category=CryptographyDeprecationWarning)

import os
import base64
import typer
import glob
import shutil
import git
import time
import json
import pkg_resources
import boto3
from typing import Optional
from argparse import ArgumentParser
from docker import DockerClient
from runhouse.common import Common
from runhouse.ssh_manager import SSHManager
from runhouse.utils.utils import ERROR_FLAG
from runhouse.config_parser import Config
from runhouse.utils.aws_utils import push_image_to_ecr, build_ecr_client
from runhouse.utils.docker_utils import file_has_changed, get_path_to_dockerfile, launch_local_docker_client, \
    build_image, generate_image_id, image_tag_name, create_dockerfile, create_or_update_docker_ignore, \
    full_ecr_tag_name
from runhouse.utils.file_utils import write_stdout_to_file, get_name_from_path, create_directory, \
    get_subdir_from_parent_dir
from runhouse.utils.validation import validate_runnable_file_path, validate_name, validate_hardware, valid_filepath, \
    validate_pem_file

# # creates an explicit Typer application, app
app = typer.Typer(add_completion=False)

# Where to create the runhouse internal subdirectory (assume it should be in the user's current directory)
RUNHOUSE_DIR = os.path.join(os.getcwd(), 'rh')

# TODO We need to inject these in a different way (can't have the user have access to the variables)
from dotenv import load_dotenv

load_dotenv()

# Create config object used for managing the read / write to the config file
cfg = Config()


def version_callback(value: bool) -> None:
    if value:
        name = 'runhouse'
        version = pkg_resources.get_distribution(name).version
        typer.echo(f"{name}=={version}")
        raise typer.Exit()


def get_hostname_from_hardware(hardware):
    """Based on mappings from hardware name to hostname IP"""
    hostname = json.loads(os.getenv('HARDWARE_TO_HOSTNAME')).get(hardware)

    if hostname is None:
        typer.echo(f"{ERROR_FLAG} host name not found for hardware {hardware}")
        raise typer.Exit(code=1)

    return hostname


def get_tag_names(use_base_image, ctx, config_kwargs, name):
    """Get the image tag and full path to image in the ecr repo"""
    image_tag = ctx.obj.image or config_kwargs.get('image_tag')

    if use_base_image:
        # use specific tag for the generic image and override any previously saved image tag
        image_tag = os.getenv('BASE_IMAGE_TAG')

    if image_tag == os.getenv('BASE_IMAGE_TAG'):
        # if using base image we give a predefined tag for the base image
        # https://hub.docker.com/r/tiangolo/python-machine-learning
        ecr_tag_name = f'tiangolo/python-machine-learning:{image_tag}'
        return image_tag, ecr_tag_name

    if image_tag is None:
        # if it hasn't yet been defined give it a random id
        random_id = generate_image_id()
        # Update the image tag with the newly generated id
        image_tag = image_tag_name(name, random_id)

    ecr_tag_name = full_ecr_tag_name(image_tag)
    return image_tag, ecr_tag_name


def does_image_exist_on_server(ssh_manager, ecr_tag_name) -> bool:
    """Check if the docker registry on the server has the image we need"""
    typer.echo(f'[1/5] Looking for existing image')
    ecr_client: boto3.client = build_ecr_client()
    token = ecr_client.get_authorization_token()
    username, password = base64.b64decode(token['authorizationData'][0]['authorizationToken']).decode().split(':')
    registry = token['authorizationData'][0]['proxyEndpoint']

    # login in via exec command
    command = f'docker login -u {username} -p {password} {registry}'
    stdout = ssh_manager.execute_command_on_remote_server(command)
    if stdout.decode('utf-8').strip() != 'Login Succeeded':
        typer.echo('Unable to login with docker credentials provided')
        raise typer.Exit(code=1)

    # check if the image exists on the local docker registry of the server
    command = f'docker inspect --type=image {ecr_tag_name}'
    stdout = ssh_manager.execute_command_on_remote_server(command)
    # if no image found the output will look like this: b'[]\n'
    return not stdout.decode('utf-8').strip() == '[]'


def run_image_on_remote_server(ssh_manager, run_cmd, ecr_tag_name, hardware, name_dir, image_exists_on_server):
    """Download the image from ecr if it doesn't exist on the server - then run it"""
    try:
        if not image_exists_on_server:
            # Pull the image to the servers local docker registry if the image doesn't exist there
            typer.echo(f"Pulling image to {hardware}")
            command = f'docker pull {ecr_tag_name}'
            stdout = ssh_manager.execute_command_on_remote_server(command)
            if not stdout:
                # if we fail to pull the image from ecr
                typer.echo(f'{ERROR_FLAG} Failed pulling image to server')
                raise typer.Exit(code=1)

        typer.echo(f'[4/5] Running image on hardware {hardware}')
        command = f'docker run {ecr_tag_name} {run_cmd}'

        # TODO may need this for .sh files - need to test
        # command = f'docker run {ecr_tag_name} /bin/bash {run_cmd}'

        stdout: bytes = ssh_manager.execute_command_on_remote_server(command)
        log_file_for_output = os.path.join(name_dir, "output.txt")
        write_stdout_to_file(json.dumps(stdout.decode("utf-8")), path_to_file=log_file_for_output)

    except Exception:
        typer.echo(f'{ERROR_FLAG} Failed to run image on {hardware}')
        raise typer.Exit(code=1)


def create_sh_file_in_dir(dir_name, file_name, text):
    with open(os.path.join(dir_name, file_name), 'w') as rsh:
        rsh.write(f'''#! /bin/sh\n{text}''')


def runnable_file_command(root_dir_for_container, file_name, formatted_args):
    """Command to be stored in the .sh file in the rh subdirectory"""
    return f"{root_dir_for_container}/{file_name} {formatted_args}"


def create_runnable_file_in_runhouse_subdir(path_to_runnable_file, file_name, root_dir_for_container, name_dir,
                                            optional_cli_args):
    """Build the internal file in the runhouse directory used for executing the code the user wants to run remotely"""
    ext = os.path.splitext(path_to_runnable_file)[1]
    formatted_args: str = ' '.join(optional_cli_args)
    cmd = runnable_file_command(root_dir_for_container, file_name, formatted_args)

    if ext not in ['.py', '.sh']:
        typer.echo(f'{ERROR_FLAG} Runhouse currently supports file types with extensions .py or .sh')
        raise typer.Exit(code=1)
    elif ext == '.py':
        cmd = f"python3 {cmd}"
    else:
        # If file is already in sh format then we are all set
        pass

    create_sh_file_in_dir(dir_name=name_dir, file_name='run.sh', text=cmd)
    return cmd


def user_rebuild_response(resp: str) -> bool:
    if resp.lower() not in ['yes', 'y', 'no', 'n']:
        typer.echo(f'{ERROR_FLAG} Invalid rebuild prompt')
        raise typer.Exit(code=1)
    return resp.lower() in ['yes', 'y']


def should_we_rebuild(ctx, optional_args, config_path, config_kwargs, path_to_dockerfile, package_time_added,
                      path_to_package) -> bool:
    """Determine if we need to rebuild the image based on the CLI arguments provided by the user + the config"""
    if not valid_filepath(config_path):
        # If the config was deleted / moved or doesn't exist at all definitely have to rebuild
        return True

    if optional_args:
        # If the user provided optional arguments to the runhouse command check whether they want to rebuild
        resp = typer.prompt('Optional args provided - would you like to rebuild? (yes / y or no / n)')
        if user_rebuild_response(resp):
            return True

    # If the user provided any options compare them to the the ones previously provided
    provided_options = ctx.obj.user_provided_args
    changed_vals = {k: provided_options[k] for k in provided_options if provided_options[k] != config_kwargs[k]
                    and provided_options[k] is not None and k in list(ctx.obj.args_to_check)}

    # If any argument(s) differs let's ask the user if they want to trigger a rebuild
    if changed_vals:
        resp = typer.prompt(f'New options provided for: {", ".join(list(changed_vals))} \n'
                            f'Would you like to rebuild? (yes / y or no / n)')
        if user_rebuild_response(resp):
            return True

    dockerfile_time_added = config_kwargs.get('dockerfile_time_added')
    if dockerfile_time_added is None or not valid_filepath(path_to_dockerfile):
        # if no dockerfile exists we need to rebuild
        return True

    # check if the user changed the dockerfile manually (if so need to trigger a rebuild)
    if file_has_changed(float(dockerfile_time_added), path_to_dockerfile):
        resp = typer.prompt('Dockerfile has been updated - would you like to rebuild? (yes / y or no / n)')
        if user_rebuild_response(resp):
            return True

    if file_has_changed(float(package_time_added), path_to_package):
        resp = typer.prompt('Package has changed - would you like to rebuild? (yes / y or no / n)')
        if user_rebuild_response(resp):
            return True

    return False


def should_we_use_base_image(path_to_parent_dir) -> bool:
    """Search for requirements file in the package directory"""
    reqs_file = next(iter(glob.glob(f'{path_to_parent_dir}/**/requirements.txt', recursive=True)), None)
    if reqs_file is None:
        typer.echo(f'No requirements.txt found in package directory - using the runhouse base image '
                   f'(tiangolo/python-machine-learning) instead')
        return True
    return False


def parse_cli_args(cli_args: list):
    """Parse the additional arguments the user provides to a given command"""
    parser = ArgumentParser()
    parser.add_argument('--package', dest='package', help='local directory or github URL', type=str)
    parser.add_argument('--hardware', dest='hardware', help='named hardware instance', type=str)
    parser.add_argument('--name', dest='name', help='name of the send', type=str)

    args = parser.parse_args(cli_args)
    return vars(args)


def bring_config_path_and_kwargs(internal_rh_name_dir, name) -> [str, dict]:
    config_path = os.path.join(internal_rh_name_dir, Config.CONFIG_FILE)
    # Try to read the config file which we will begin to populate with the params the user provided
    config_kwargs = cfg.bring_config_kwargs(config_path, name)
    return config_path, config_kwargs


@app.command(context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def run(ctx: typer.Context):
    """Run code for given name based on provided configurations"""
    start = time.time()

    optional_cli_args: list = ctx.args
    parsed_cli_args: dict = parse_cli_args(optional_cli_args)

    # ADD CTX PARAMS HERE
    send = ctx.obj.send

    file_to_run = ctx.obj.file
    # If file to run is provided we will need these values
    path_to_runnable_file = None
    run_cmd = None
    root_dir_name_for_container = None
    full_path_to_package = ''

    name = ctx.obj.name or parsed_cli_args.get('name')
    if name is None:
        typer.echo('No name provided for the run')
        raise typer.Exit(code=1)

    # Make sure we have the main rh directory in the local filesystem
    create_directory(RUNHOUSE_DIR)

    if send:
        func_type = 'send'
        internal_rh_name_dir = os.path.join(RUNHOUSE_DIR, 'sends', name)
        create_directory(internal_rh_name_dir)

        package = parsed_cli_args.get('package')
        config_path, config_kwargs = bring_config_path_and_kwargs(internal_rh_name_dir, name)

        if package.endswith('.git'):
            # clone the package locally
            typer.echo(f'Cloning github URL as package for the send ({package})')
            try:
                git.Git(internal_rh_name_dir).clone(package)
            except git.GitCommandError:
                # clone either failed or already exists locally
                # TODO differentiate between failed clone vs. directory already exists error
                pass

            # path to the repo will be in the child directory of the send's folder
            full_path_to_package = get_subdir_from_parent_dir(internal_rh_name_dir)

        else:
            # Using a local directory as a package
            path_to_parent = os.path.abspath(package)
            full_path_to_package = path_to_parent
            if not valid_filepath(full_path_to_package):
                typer.echo(f'Package with path {full_path_to_package} not found')
                raise typer.Exit(code=1)

            # package refers to local directory
            typer.echo(f'Using local directory {full_path_to_package} as package for the send')
            # copy the package to the send's directory
            shutil.copytree(full_path_to_package, internal_rh_name_dir, dirs_exist_ok=True)

    elif file_to_run:
        func_type = 'run'
        internal_rh_name_dir = os.path.join(RUNHOUSE_DIR, name)
        create_directory(internal_rh_name_dir)

        config_path, config_kwargs = bring_config_path_and_kwargs(internal_rh_name_dir, name)

        # if we are given a specific file to run (assume this has some sort of main entrypoint)
        # Update the path to the runnable file with one defined in the config if it exists
        path_to_runnable_file = file_to_run or config_kwargs.get('file')
        validate_runnable_file_path(path_to_runnable_file)
        file_name = get_name_from_path(path_to_runnable_file)

        # Root directory where the file is being run - this is needed for running the file in the container
        root_dir_name_for_container: str = os.path.dirname(path_to_runnable_file)

        # Create a runnable file with the arguments provided in the internal runhouse subdirectory
        run_cmd: str = create_runnable_file_in_runhouse_subdir(path_to_runnable_file, file_name,
                                                               root_dir_name_for_container,
                                                               internal_rh_name_dir, optional_cli_args)
    else:
        typer.echo('No send or file to run provided')
        raise typer.Exit(code=1)

    package_time_added = float(config_kwargs.get('package_time_added', 0.0))

    dockerfile: str = get_path_to_dockerfile(full_path_to_package, config_kwargs, ctx)

    # Check whether we need to rebuild the image or not
    rebuild: bool = should_we_rebuild(ctx=ctx, optional_args=optional_cli_args,
                                      config_path=config_path, config_kwargs=config_kwargs,
                                      path_to_dockerfile=dockerfile, package_time_added=package_time_added,
                                      path_to_package=full_path_to_package)

    hardware = ctx.obj.hardware or config_kwargs.get('hardware', os.getenv('DEFAULT_HARDWARE'))
    validate_hardware(hardware)

    # Check whether we use an image based on the user's requirements or a base image
    use_base_image: bool = should_we_use_base_image(full_path_to_package)

    # Make sure we ignore the runhouse directory when building the image
    create_or_update_docker_ignore(full_path_to_package)

    # TODO if using base image save to a public repository?
    image_tag, ecr_tag_name = get_tag_names(use_base_image, ctx, config_kwargs, name)

    # Check if we have the pem file needed to connect to the server
    path_to_pem = validate_pem_file()

    # Connect to the server, then check if the image exists on the server
    hostname = get_hostname_from_hardware(hardware)
    ssh_manager = SSHManager(hostname=hostname, path_to_pem=path_to_pem)
    ssh_manager.connect_to_server()

    image_exists_on_server = does_image_exist_on_server(ssh_manager, ecr_tag_name)
    if not image_exists_on_server or rebuild:
        if not use_base_image:
            # We are in rebuild mode and have enough info not to rely on a pre-existing image
            if not valid_filepath(dockerfile) or file_has_changed(package_time_added, full_path_to_package):
                # if dockerfile still does not exist (ex: user deleted it) or package was updated
                typer.echo(f'[2/5] Building dockerfile for {func_type}')
                dockerfile = create_dockerfile(full_path_to_package)

            docker_client: DockerClient = launch_local_docker_client()
            # Build the image locally before pushing to ecr
            typer.echo(f'[3/5] Building image for {func_type}')
            image_obj = build_image(dockerfile, docker_client, image_tag, full_path_to_package, hardware)
            push_image_to_ecr(docker_client, image_obj, tag_name=image_tag)
        else:
            # let's use a predefined base image that we already have on the server
            typer.echo(f'[2/5] Loading runhouse base image for {func_type}')
            typer.echo(f'[3/5] Running runhouse base image with existing dockerfile')

    if send:
        typer.echo(f'[5/5] Finished building {func_type}, updating config')

    if file_to_run:
        # Once we've made sure we have the image we need in ecr we can run it on the server
        run_image_on_remote_server(ssh_manager, run_cmd, ecr_tag_name, hardware, full_path_to_package,
                                   image_exists_on_server)
        typer.echo(f'[5/5] Finished running {func_type}, updating config')

    # create or update the config if it doesn't already exist
    cfg.create_or_update_config_file(internal_rh_name_dir, path=full_path_to_package, file=path_to_runnable_file,
                                     name=name, hardware=hardware, dockerfile=dockerfile, image_tag=image_tag,
                                     rebuild=rebuild, config_kwargs=config_kwargs,
                                     container_root=root_dir_name_for_container)

    end = time.time()
    typer.echo(f'{func_type} completed in {int(end - start)} seconds')

    raise typer.Exit()


@app.callback()
def common(ctx: typer.Context,
           dockerfile: str = typer.Option(None, '--dockerfile', '-d', help='path to existing dockerfile'),
           file: str = typer.Option(None, '--file', '-f', help='path to specific file to run '
                                                               '(contained in the path provided'),
           hardware: str = typer.Option(None, '--hardware', '-h', help='desired hardware'),
           image: str = typer.Option(None, '--image', '-i', help='image tag of existing local docker image'),
           name: str = typer.Option(None, '--name', '-n', help='name of the existing run or new run'),
           path: str = typer.Option(None, '--path', '-p', help='path to directory or github URL to be packaged '
                                                               'as part of the run'),
           ssh: bool = typer.Option(None, '--ssh', '-ssh', help='run code in interactive mode'),
           send: bool = typer.Option(None, '--send', '-s', help='create a serverless endpoint'),
           version: Optional[bool] = typer.Option(None, '--version', '-v', callback=version_callback,
                                                  help='current package version')):
    """Welcome to Runhouse! Here's what you need to get started"""
    ctx.obj = Common(name, hardware, dockerfile, file, image, ssh, path, send)
