"""This module provides the runhouse CLI."""
import warnings

# Suppress warnings when running commands
from cryptography.utils import CryptographyDeprecationWarning
warnings.filterwarnings("ignore", category=CryptographyDeprecationWarning)

import os
import base64
import typer
import tarfile
import time
import json
import pkg_resources
import boto3
from typing import Optional
from docker import DockerClient
from runhouse.common import Common
from runhouse.ssh_manager import SSHManager
from runhouse.utils.utils import ERROR_FLAG
from runhouse.config_parser import Config
from runhouse.utils.aws_utils import push_image_to_ecr, build_ecr_client, upload_file_to_s3
from runhouse.utils.docker_utils import file_has_changed, get_path_to_dockerfile, launch_local_docker_client, \
    build_image, generate_image_id, image_tag_name, create_dockerfile, create_or_update_docker_ignore, \
    full_ecr_tag_name
from runhouse.utils.file_utils import create_name_for_folder, delete_directory, create_directories, \
    write_stdout_to_file, get_name_from_path, delete_file
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


def rename_callback(name: str) -> None:
    name_dir = os.path.join(RUNHOUSE_DIR, name)
    if not valid_filepath(name_dir):
        typer.echo(f'{ERROR_FLAG} {name} does not exist')
        return None

    # TODO rather than a prompt allow the user to specify in the same command (similar to mv <src> <dest>)
    new_dir_name = typer.prompt("Please enter the new name")
    renamed_dir = os.path.join(RUNHOUSE_DIR, new_dir_name)
    if any([name for name in os.listdir(RUNHOUSE_DIR) if name == new_dir_name]):
        typer.echo(f'{ERROR_FLAG} {new_dir_name} already exists, cannot rename')
        raise typer.Exit(code=1)

    validate_name(new_dir_name)
    os.rename(name_dir, renamed_dir)

    # TODO we also need to update the image tag
    cfg.create_or_update_config_file(renamed_dir, name=new_dir_name, rename=True)


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
    # Path to requirements is assumed to be in the parent dir
    path_to_reqs = os.path.join(path_to_parent_dir, 'requirements.txt')
    if not valid_filepath(path_to_reqs):
        typer.echo(f'No requirements.txt found in parent directory - using the runhouse base image '
                   f'(tiangolo/python-machine-learning) instead')
        return True
    return False


def build_compressed_package(path_to_parent_dir, package_name):
    package_file_name = f"{os.path.join(path_to_parent_dir, package_name)}.tar.gz"
    with tarfile.open(package_file_name, mode='w:gz') as archive:
        archive.add(path_to_parent_dir)
    return package_file_name


@app.command(context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def run(ctx: typer.Context):
    """Run code for given name based on provided configurations"""
    # TODO this is a long ass function - let's clean it up
    start = time.time()

    rename = ctx.obj.rename
    if rename:
        # For renaming not running anything - just a directory change name and config changes
        # TODO we also need to change the image tag
        rename_callback(rename)
        raise typer.Exit()

    # If user chooses a one-time run we won't be saving anything new to the user's runhouse directory
    anon = ctx.obj.anon
    name = 'anon' if anon else create_name_for_folder(name=ctx.obj.name)

    # Directory for the name which will sit in the runhouse folder
    internal_rh_name_dir = os.path.join(RUNHOUSE_DIR, name)
    config_path = os.path.join(internal_rh_name_dir, Config.CONFIG_FILE)
    # Try to read the config file which we will begin to populate with the params the user provided
    config_kwargs: dict = cfg.bring_config_kwargs(config_path, name)

    # Make sure hardware specs are valid
    hardware = ctx.obj.hardware or config_kwargs.get('hardware', os.getenv('DEFAULT_HARDWARE'))
    validate_hardware(hardware)

    # Generate the path to the parent dir for building the image
    path_to_parent_dir = ctx.obj.path or config_kwargs.get('external_package', os.getcwd())
    package_time_added = float(config_kwargs.get('package_time_added', 0.0))

    # TODO save this compressed tar.gz to s3, then load into their image
    package_name = get_name_from_path(path_to_parent_dir)
    package_tar = build_compressed_package(path_to_parent_dir, package_name)

    # save this tar to s3 bucket
    upload_file_to_s3(package_tar, package_name)

    dockerfile: str = get_path_to_dockerfile(path_to_parent_dir, config_kwargs, ctx)

    # Additional args used for running the file (separate from the predefined CLI options)
    optional_cli_args: list = ctx.args
    # Check whether we need to rebuild the image or not
    rebuild: bool = should_we_rebuild(ctx=ctx, optional_args=optional_cli_args,
                                      config_path=config_path, config_kwargs=config_kwargs,
                                      path_to_dockerfile=dockerfile, package_time_added=package_time_added,
                                      path_to_package=path_to_parent_dir)

    # Check whether we use an image based on the user's requirements or a base image
    use_base_image: bool = should_we_use_base_image(path_to_parent_dir)

    # make sure we have the relevant directories created on the user's file system
    # create the main runhouse dir and the subdir of the named uri
    create_directories(dir_names=[RUNHOUSE_DIR, internal_rh_name_dir])

    # Make sure we ignore the runhouse directory when building the image (it will sit in the root dir as well)
    create_or_update_docker_ignore(path_to_parent_dir)

    # Update the path to the runnable file with one defined in the config if it exists
    path_to_runnable_file = ctx.obj.file or config_kwargs.get('file')
    validate_runnable_file_path(path_to_runnable_file)

    file_name = get_name_from_path(path_to_runnable_file)
    # Root directory where the file is being run - this is needed for running the file in the container
    root_dir_name_for_container = os.path.dirname(path_to_runnable_file)

    # Create a runnable file with the arguments provided in the internal runhouse subdirectory
    run_cmd: str = create_runnable_file_in_runhouse_subdir(path_to_runnable_file, file_name,
                                                           root_dir_name_for_container,
                                                           internal_rh_name_dir, optional_cli_args)

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
            if not valid_filepath(dockerfile) or file_has_changed(package_time_added, path_to_parent_dir):
                # if dockerfile still does not exist (ex: user deleted it) or package was updated
                typer.echo('[2/5] Building dockerfile')
                dockerfile = create_dockerfile(path_to_parent_dir, package_tar)

            docker_client: DockerClient = launch_local_docker_client()
            # Build the image locally before pushing to ecr
            typer.echo(f'[3/5] Building image')
            image_obj = build_image(dockerfile, docker_client, name, image_tag, path_to_parent_dir, hardware,
                                    package_tar)
            push_image_to_ecr(docker_client, image_obj, tag_name=image_tag)
        else:
            # let's use a predefined base image that we already have on the server
            typer.echo('[2/5] Loading runhouse base image')
            typer.echo('[3/5] Running runhouse base image with existing dockerfile')

    # Once we've made sure we have the image we need in ecr we can run it on the server
    run_image_on_remote_server(ssh_manager, run_cmd, ecr_tag_name, hardware, internal_rh_name_dir, image_exists_on_server)
    typer.echo(f'[5/5] Finished running')

    # create or update the config if it doesn't already exist
    cfg.create_or_update_config_file(internal_rh_name_dir, path=path_to_parent_dir, file=path_to_runnable_file,
                                     name=name, hardware=hardware, dockerfile=dockerfile, image_tag=image_tag,
                                     rebuild=rebuild, config_kwargs=config_kwargs,
                                     container_root=root_dir_name_for_container, package_tar=package_tar)
    if anon:
        # delete the directory which we needed to create for the one-time run
        delete_directory(internal_rh_name_dir)

    end = time.time()
    typer.echo(f'Run completed in {int(end - start)} seconds')

    raise typer.Exit()


@app.callback()
def common(ctx: typer.Context,
           anon: bool = typer.Option(None, '--anon', '-a',
                                     help="anonymous/one-time run (run will not be named and config won't be created)"),
           dockerfile: str = typer.Option(None, '--dockerfile', '-d', help='path to existing dockerfile'),
           file: str = typer.Option(None, '--file', '-f', help='path to specific file to run '
                                                               '(contained in the path provided'),
           hardware: str = typer.Option(None, '--hardware', '-h', help='desired hardware'),
           image: str = typer.Option(None, '--image', '-i', help='image tag of existing local docker image'),
           name: str = typer.Option(None, '--name', '-n', help='name of the existing run or new run'),
           path: str = typer.Option(None, '--path', '-p', help='path to directory or github URL to be packaged '
                                                               'as part of the run'),
           rename: str = typer.Option(None, '--rename', '-r', help='rename existing URI'),
           shell: bool = typer.Option(None, '--shell', '-s', help='run code in interactive mode'),
           version: Optional[bool] = typer.Option(None, '--version', '-v', callback=version_callback,
                                                  help='current package version')):
    """Welcome to Runhouse! Here's what you need to get started"""
    ctx.obj = Common(name, hardware, dockerfile, file, image, shell, path, anon, rename)
