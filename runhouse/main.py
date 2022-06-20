"""This module provides the runhouse CLI."""
import warnings

# Suppress warnings when running commands
from cryptography.utils import CryptographyDeprecationWarning
warnings.filterwarnings("ignore", category=CryptographyDeprecationWarning)

import os
import typer
import base64
import pkg_resources
from typing import Optional
from docker import DockerClient
from pathlib import Path
from runhouse.common import Common
from runhouse.shell_handler import ShellHandler
from runhouse.ssh_manager import SSHManager
from runhouse.process_commands import process_cmd_commands
from runhouse.utils.utils import ERROR_FLAG
from runhouse.config_parser import Config
from runhouse.utils.deploy_to_aws import push_image_to_ecr, build_ecr_client
from runhouse.utils.docker_utils import dockerfile_has_changed, get_path_to_dockerfile, launch_local_docker_client, \
    bring_image_from_docker_client, build_image, generate_image_id, image_tag_name, create_dockerfile
from runhouse.utils.file_utils import create_name_for_folder, delete_directory, create_directories, \
    copy_file_to_directory
from runhouse.utils.validation import validate_runnable_file_path, validate_name, validate_hardware, valid_filepath

# # creates an explicit Typer application, app
app = typer.Typer(add_completion=False)

RUNHOUSE_DIR = os.path.join(os.getcwd(), Path(__file__).parent.parent.absolute(), 'runhouse')

# Mapping each hardware option to its elastic IP
# TODO move this shit into config file
HARDWARE_TO_HOSTNAME = {"rh_1_gpu": "3.136.181.23", "rh_8_gpu": "3.136.181.23", "rh_16_gpu": "3.136.181.23"}
ECR_URI = '675161046833.dkr.ecr.us-east-1.amazonaws.com/runhouse-demo'
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
        return None

    validate_name(new_dir_name)
    os.rename(name_dir, renamed_dir)
    cfg.create_or_update_config_file(renamed_dir, name=new_dir_name, rename=True)


def get_hostname_from_hardware(hardware):
    """Based on mappings from hardware name to hostname IP"""
    hostname = HARDWARE_TO_HOSTNAME.get(hardware)

    if hostname is None:
        typer.echo(f"{ERROR_FLAG} host name not found for hardware {hardware}")
        typer.Exit(code=1)

    return hostname


def open_bash_on_remote_server(hardware):
    """Open bash on remote env in user """
    host = get_hostname_from_hardware(hardware)
    typer.echo(f"Opening shell with {hardware} on host {host}\n")
    sh = ShellHandler(host=host)
    process_cmd_commands(sh)


def run_image_on_remote_server(tag_name, hardware):
    """Download the image from ecr if it doesn't exist on the server - then run it"""
    try:
        # Connect/ssh to an instance
        hostname = get_hostname_from_hardware(hardware)

        path_to_pem = os.getenv('PATH_TO_PEM')
        if path_to_pem is None:
            resp = typer.prompt('Please specify path to your runhouse pem file (or add as env variable "PATH_TO_PEM")')
            path_to_pem = Path(resp)
            if not valid_filepath(path_to_pem):
                typer.echo('Invalid file path to pem')
                raise typer.Exit(code=1)

        sm = SSHManager(hostname=hostname, path_to_pem=path_to_pem)
        sm.connect_to_server()

        # Copy the image to namespace folder on the server
        ecr_client = build_ecr_client()
        token = ecr_client.get_authorization_token()

        username, password = base64.b64decode(token['authorizationData'][0]['authorizationToken']).decode().split(':')
        registry = token['authorizationData'][0]['proxyEndpoint']

        # login in via the docker sdk doesnt work so we're gonna go with this workaround
        command = 'docker login -u %s -p %s %s' % (username, password, registry)
        sm.execute_command_on_remote_server(command)

        typer.echo(f'[1/3] Retrieving image')
        # Check if the image exists on the server's local image registry
        command = f'docker image inspect {ECR_URI}:{tag_name}'
        stdout = sm.execute_command_on_remote_server(command)
        if not stdout:
            # Pull the image to the servers local docker registry if the image doesn't exist
            typer.echo("Pulling existing image")
            command = f'docker pull {ECR_URI}:{tag_name}'
            sm.execute_command_on_remote_server(command)

        # Run the image on the server
        typer.echo(f'[2/3] Running image on hardware {hardware}')
        command = f'docker run {ECR_URI}:{tag_name} /bin/bash -c "chmod +x ./run.sh; ./run.sh" '
        sm.execute_command_on_remote_server(command)

    except Exception as e:
        typer.echo(f'{ERROR_FLAG} Unable to run image on remote server', e)
        raise typer.Exit(code=1)


def build_path_to_parent_dir(ctx, config_kwargs):
    # Check if user has explicitly provided a path to the parent directory
    path_to_parent_dir = ctx.obj.path or config_kwargs.get('path')
    if path_to_parent_dir:
        return Path(path_to_parent_dir)

    # If the user gave us no indication of where the parent directory is assume its in the parent folder
    # (based on where the CLI command is being run)
    return Path(RUNHOUSE_DIR).parent.absolute()


def create_sh_file_in_dir(dir_name, file_name, text):
    with open(os.path.join(dir_name, file_name), 'w') as rsh:
        rsh.write(f'''#! /bin/sh\n{text}''')


def create_runnable_file_in_runhouse_subdir(path_to_runnable_file, name_dir, optional_cli_args):
    """Build the internal file in the runhouse directory used for executing the code the user wants to run remotely"""
    ext = os.path.splitext(path_to_runnable_file)[1]
    file_name = os.path.basename(path_to_runnable_file)
    formatted_args: str = ' '.join(optional_cli_args)
    path_to_file_from_subdir = os.path.join(Path(__file__).parents[0], file_name)

    if ext not in ['.py', '.sh']:
        typer.echo(f'{ERROR_FLAG} Runhouse currently supports file types with extensions .py or .sh')
        raise typer.Exit(code=1)
    elif ext == '.py':
        cmd = f"python3 {path_to_file_from_subdir} {formatted_args}"
    else:
        cmd = f"{path_to_file_from_subdir} {formatted_args}"

    create_sh_file_in_dir(dir_name=name_dir, file_name='run.sh', text=cmd)


def user_rebuild_response(resp: str) -> bool:
    if resp.lower() not in ['yes', 'y', 'no', 'n']:
        typer.echo(f'{ERROR_FLAG} Invalid rebuild prompt')
        raise typer.Exit(code=1)
    return resp.lower() in ['yes', 'y']


def should_we_rebuild(ctx, optional_args, config_path, config_kwargs, path_to_dockerfile) -> bool:
    """Determine if we need to rebuild the image based on the CLI arguments provided by the user + the config"""
    if not valid_filepath(config_path):
        # If there is no config in the name directory we definitely have to rebuild
        return True

    if optional_args:
        # If the user provided optional arguments to the runhouse command check whether they want to rebuild
        resp = typer.prompt('Optional args provided - would you like to rebuild? (Yes / Y or No / N)')
        if user_rebuild_response(resp):
            return True

    # If the user provided any options compare them to the the ones previously provided
    provided_options = ctx.obj.user_provided_args
    changed_vals = {k: provided_options[k] for k in provided_options if provided_options[k] != config_kwargs[k]
                    and provided_options[k] is not None and k in list(ctx.obj.args_to_check)}

    # If any argument(s) differs let's ask the user if they want to trigger a rebuild
    if changed_vals:
        resp = typer.prompt(f'New options provided for: {", ".join(list(changed_vals))} \n'
                            f'Would you like to rebuild? (Yes / Y or No / N)')
        if user_rebuild_response(resp):
            return True

    dockerfile_time_added = config_kwargs.get('dockerfile_time_added')
    if dockerfile_time_added is None or not valid_filepath(path_to_dockerfile):
        # if no dockerfile exists we need to rebuild
        return True

    # check if the user changed the dockerfile manually (if so need to trigger a rebuild)
    rebuild = False
    if dockerfile_has_changed(float(dockerfile_time_added), path_to_dockerfile):
        resp = typer.prompt('Dockerfile has been updated - would you like to rebuild? (Yes / Y or No / N)')
        rebuild = user_rebuild_response(resp)

    return rebuild


@app.command(context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def run(ctx: typer.Context):
    """Run code for given name based on provided configurations"""
    if ctx.obj.rename:
        # For renaming not running anything - just a directory change name and config changes
        rename_callback(ctx.obj.rename)
        raise typer.Exit()

    # If user chooses a one-time run we won't be saving anything new to the user's runhouse directory
    anon = ctx.obj.anon
    name = 'anon' if anon else create_name_for_folder(name=ctx.obj.name)

    # Directory for the name which will sit in the runhouse folder
    name_dir = os.path.join(RUNHOUSE_DIR, name)
    config_path = os.path.join(name_dir, Config.CONFIG_FILE)

    # Try to read the config file which we will begin to populate with the params the user provided
    config_kwargs: dict = cfg.bring_config_kwargs(config_path, name)

    # Update the path to the runnable file with one defined in the config if it exists
    path_to_runnable_file = ctx.obj.file or config_kwargs.get('file')

    # Make sure hardware specs are valid
    hardware = ctx.obj.hardware or config_kwargs.get('hardware')
    validate_hardware(hardware, hardware_to_hostname=HARDWARE_TO_HOSTNAME)

    # Generate the path to where the dockerfile should live
    path_to_parent_dir: Path = build_path_to_parent_dir(ctx, config_kwargs)
    dockerfile: str = get_path_to_dockerfile(path_to_parent_dir, config_kwargs, ctx)
    images_dir: str = os.path.join(name_dir, "images")

    # Additional args used for running the file (separate from the predefined CLI options)
    optional_cli_args: list = ctx.args
    # Check whether we need to rebuild the image or not
    rebuild: bool = should_we_rebuild(ctx=ctx, optional_args=optional_cli_args,
                                      config_path=config_path, config_kwargs=config_kwargs,
                                      path_to_dockerfile=dockerfile)
    # Path to requirements is assumed to be in the parent dir
    path_to_reqs = os.path.join(path_to_parent_dir, 'requirements.txt')
    if not valid_filepath(path_to_reqs):
        typer.echo(f'{ERROR_FLAG} No requirements.txt found in parent directory - please add before continuing')
        raise typer.Exit(code=1)

    # make sure we have the relevant directories created on the user's file system
    # create the main runhouse dir, the subdir of the named uri, and the images dir
    create_directories([RUNHOUSE_DIR, name_dir, images_dir])

    # TODO is there a way around this? You can only copy files that are located within the local source tree
    copy_file_to_directory(path_to_reqs, os.path.join(name_dir, "requirements.txt"))

    # Check if user provided the name of the file to be executed
    validate_runnable_file_path(path_to_runnable_file)

    # Create a runnable file with the arguments provided in the internal runhouse subdirectory
    create_runnable_file_in_runhouse_subdir(path_to_runnable_file, name_dir, optional_cli_args)
    image_id = ctx.obj.image or config_kwargs.get('image_id')

    docker_client: DockerClient = launch_local_docker_client()
    # Try loading the local image if it exists
    image_obj = bring_image_from_docker_client(docker_client, image_id)
    tag_name = image_tag_name(name, image_id)
    if image_obj is None or rebuild:
        # If no image exists or we were instructed to rebuild anyways
        if image_id is None:
            # if it hasn't yet been defined give it a random id
            image_id = generate_image_id()
            # Update the tag name if we are rebuilding the image
            tag_name = image_tag_name(name, image_id)

        if not valid_filepath(dockerfile):
            # if dockerfile still does not exist we need to build it
            typer.echo('Building Dockerfile')
            # TODO we want the whole root dir contents in the dockerfile, but only the specific runhouse subdir
            dockerfile = create_dockerfile(path_to_parent_dir)

        image_obj = build_image(dockerfile, docker_client, name, tag_name, name_dir, hardware)

        # save the image directly to a remote repository (ECR) and then pull within ec2
        push_image_to_ecr(docker_client, image_obj, tag_name=tag_name)

    # Copy the image to remote server and run it there
    run_image_on_remote_server(tag_name, hardware)
    typer.echo(f'[3/3] Finished running')

    # set up the config so we have it for the next run
    cfg.create_or_update_config_file(name_dir, path=path_to_parent_dir, file=path_to_runnable_file, name=name,
                                     hardware=hardware, dockerfile=dockerfile, image_id=image_id,
                                     rebuild=rebuild, config_kwargs=config_kwargs)
    if anon:
        # delete the directory which we needed to create for the one-time run
        delete_directory(name_dir)

    raise typer.Exit()


@app.callback()
def common(ctx: typer.Context,
           anon: bool = typer.Option(None, '--anon', '-a',
                                     help="anonymous/one-time run (run will not be named and config won't be created)"),
           dockerfile: str = typer.Option(None, '--dockerfile', '-d', help='path to existing dockerfile'),
           file: str = typer.Option(None, '--file', '-f', help='Specific file path to be run'),
           hardware: str = typer.Option(None, '--hardware', '-h', help='desired hardware'),
           image: str = typer.Option(None, '--image', '-i', help='image id of existing local docker image'),
           name: str = typer.Option(None, '--name', '-n', help='name your microservice / URI'),
           path: str = typer.Option(None, '--path', '-p',
                                    help='Path to parent directory to be packaged and registered as a URI'),
           rename: str = typer.Option(None, '--rename', '-r', help='rename existing URI'),
           shell: bool = typer.Option(None, '--shell', '-s', help='run code in interactive mode'),
           version: Optional[bool] = typer.Option(None, '--version', '-v', callback=version_callback,
                                                  help='current package version')):
    """Welcome to Runhouse! Here's what you need to get started"""
    ctx.obj = Common(name, hardware, dockerfile, file, image, shell, path, anon, rename)
