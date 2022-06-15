"""This module provides the runhouse CLI."""
import warnings
from cryptography.utils import CryptographyDeprecationWarning

# Suppress annoying warnings when running the CLI
warnings.filterwarnings("ignore", category=CryptographyDeprecationWarning)

import os
import json
import logging
from dataclasses import dataclass
from configparser import ConfigParser
from pathlib import Path
from typing import Optional
import docker
import typer
from runhouse import __app_name__, __version__
from runhouse.shell_handler import ShellHandler
from runhouse.ssh_manager import SSHManager
from runhouse.process_commands import process_cmd_commands
from runhouse.utils.utils import save_to_file, read_file, valid_filepath, random_string_generator, create_directory

# For now load from .env
from dotenv import load_dotenv
load_dotenv()

# # creates an explicit Typer application, app
app = typer.Typer(add_completion=False)

logger = logging.getLogger(__name__)

RUNHOUSE_DIR = os.path.join(os.getcwd(), Path(__file__).parent, 'runhouse')
# map each hardware option to its IP / host
CONFIG_FILE = 'config.ini'
MAIN_CONF_HEADER = 'main'
DOCKER_HEADER = 'docker'
HARDWARE_TO_HOSTNAME = json.loads(os.getenv('HARDWARE_TO_HOSTNAME'))


def version_callback(value: bool) -> None:
    if value:
        typer.echo(f"{__app_name__}=={__version__}")
        raise typer.Exit()


def rename_callback(value: str) -> None:
    # TODO will prob need some form of DB to really implement this
    pass


def validate_hardware(hardware):
    """Throw an error for invalid hardware specs"""
    if hardware not in list(HARDWARE_TO_HOSTNAME):
        typer.echo(f"invalid hardware specification")
        typer.echo(f"Please choose from the following options: {list(HARDWARE_TO_HOSTNAME)}")
        raise typer.Exit(code=1)


def get_hostname_from_hardware(hardware):
    hostname = HARDWARE_TO_HOSTNAME.get(hardware)

    if hostname is not None:
        return hostname

    typer.echo(f"host name not found for hardware {hardware}")
    typer.Exit(code=1)


def open_bash_on_remote_server(hardware):
    host = get_hostname_from_hardware(hardware)
    typer.echo(f"Opening shell with {hardware}")
    sh = ShellHandler(host=host)
    process_cmd_commands(sh)


def run_image_on_remote_server(path, hardware):
    try:
        # Connect/ssh to an instance
        hostname = get_hostname_from_hardware(hardware)
        sm = SSHManager(hostname=hostname)
        ftp_client = sm.create_ftp_client()

        sm.copy_file_to_remote_server(ftp_client=ftp_client, filepath=path)

        # Execute the file after connecting/ssh to the instance
        stdin, stdout, stderr = sm.client.exec_command(f"""cat <<EOF | docker exec --interactive {path} sh
                                                        cd /var/log
                                                        tar -cv ./file.log
                                                        EOF""")
        stdout = stdout.readlines()

        stdin.close()
        # close the client connection once the job is done
        sm.client.close()

    except Exception as e:
        logger.error(f'Unable to run image {path} on remote server: {e}')
        raise typer.Exit(code=1)


def create_directories_for_name(name_dir):
    # First make sure we have the parent "runhouse" dir
    create_directory(RUNHOUSE_DIR)

    # create the subdir of the named uri
    create_directory(name_dir)


def create_or_update_config_file(directory, **kwargs):
    config_path = os.path.join(directory, CONFIG_FILE)

    config = ConfigParser()
    config.read('config.ini')
    config.add_section(MAIN_CONF_HEADER)
    config.add_section(DOCKER_HEADER)

    user_params = kwargs.get('user_params')
    if isinstance(user_params, dict):
        user_params = json.dumps(user_params)

    config.set(MAIN_CONF_HEADER, 'name', kwargs.get('name'))
    config.set(MAIN_CONF_HEADER, 'hardware', kwargs.get('hardware'))
    config.set(MAIN_CONF_HEADER, 'params', user_params)
    config.set(MAIN_CONF_HEADER, 'path', directory)
    config.set(DOCKER_HEADER, 'dockerfile', kwargs.get('dockerfile'))

    try:
        with open(config_path, 'w') as f:
            config.write(f)
    except:
        typer.echo('Unable to save config file')
        raise typer.Exit(code=1)


def read_config_file(config_path):
    config = ConfigParser()
    config.read(config_path)

    # read values from file
    dockerfile = config.get(DOCKER_HEADER, 'dockerfile')
    name = config.get(MAIN_CONF_HEADER, 'name')
    hardware = config.get(MAIN_CONF_HEADER, 'hardware')
    path = config.get(MAIN_CONF_HEADER, 'path')
    params = config.get(MAIN_CONF_HEADER, 'params')

    return {'dockerfile': dockerfile, 'name': name, 'hardware': hardware, 'params': params, 'path': path}


def build_context_params(all_args):
    res = {}
    # get indices of all strings with "-" which indicates it is a key
    param_indices = [i for i, s in enumerate(all_args) if '-' in s]
    if not param_indices:
        typer.echo('Invalid format: params must be denoted with "-" or "--"')
        raise typer.Exit(code=1)

    for idx, arg in enumerate(all_args):
        if idx in param_indices:
            key = arg.replace("-", "")
            res.setdefault(key, "")
        else:
            # the string following the arg provided with a '--' will be treated as its value
            res[key] = arg
    return res


def build_and_save_image(image, path_to_image, dockerfile, docker_client, dir_path, name, name_dir,
                         user_params, hardware):
    if not image:
        # if no image url has been provided we have some work to do
        # Need to build the image based on dockerfile provided, or if that isn't provided first build the dockerfile
        path_to_reqs = os.path.join(dir_path, "requirements.txt")
        if not dockerfile:
            # Check for requirements txt
            if not valid_filepath(path_to_reqs):
                typer.echo(f'No requirements.txt found for {name}, please add before continuing')
                raise typer.Exit(code=1)
            else:
                # TODO create the dockerfile
                typer.echo('Building Dockerfile with provided requirements')

        docker_client.images.build(path=name_dir, dockerfile=dockerfile, tag=name,
                                   labels={'hardware': hardware, 'params': json.dumps(user_params)})
        typer.echo(f'[1/4] Finished building image for {name}')

    else:
        # if image exists then save it
        save_image_to_tar(image, path_to_image)
        typer.echo(f"[1/4] Successfully loaded image {image.tags} with labels: {image.labels}")


def bring_image_from_docker_client(docker_client, image_id):
    if image_id is None:
        # We may not have any image at this stage
        return None
    try:
        image = docker_client.images.get(image_id)
    except docker.errors.ImageNotFound:
        # if the image doesn't exist
        image = None
    except docker.errors.APIError:
        typer.echo('Error with docker client retrieving image')
        raise typer.Exit(code=1)
    return image


def save_image_to_tar(image, path_to_image):
    if valid_filepath(path_to_image):
        # if already saved by user we're good
        return

    f = open(path_to_image, 'wb')
    for chunk in image.save(chunk_size=2097152, named=False):
        f.write(chunk)
    f.close()


def get_path_to_dockerfile(dir_path, config_kwargs, ctx):
    path_to_dockerfile = os.path.join(dir_path, "Dockerfile")
    if valid_filepath(path_to_dockerfile):
        return path_to_dockerfile
    # if the dockerfile doesn't yet exist in filesystem try the user CLI params or the config file
    return ctx.obj.dockerfile or config_kwargs.get('dockerfile', '')


def bring_config_kwargs(config_path, name, name_dir):
    if not valid_filepath(config_path):
        # If we don't have a config for this name yet define the initial default values
        return {'name': name, 'path': name_dir, 'hardware': os.getenv('DEFAULT_HARDWARE')}
    # take from the config that already exists
    return read_config_file(config_path)


def validate_and_create_name_folder(name):
    if name is None:
        # if user did not provide a namespace for this run we'll make one up
        # TODO use real words (heroku style)
        name = random_string_generator().lower()
        typer.echo(f'Creating new runhouse URI with name: {name}')
        name_dir = os.path.join(RUNHOUSE_DIR, name)
        # if the name does not exist then we need to first create the relevant directories for it
        create_directories_for_name(name_dir)
    else:
        # For the purposes of building images (and using it as a tag), make sure it is always lower
        name = name.lower()
        # Since this name already exists confirm its associated folder exists in the file system
        name_dir = os.path.join(RUNHOUSE_DIR, name)
        if not valid_filepath(name_dir):
            typer.echo(f'Directory in the runhouse folder for {name} does not exist!')
            raise typer.Exit(code=1)

    return name, name_dir


@dataclass
class Common:
    name: str
    hardware: str
    dockerfile: str
    image: str
    shell: bool
    path: str


@app.command(context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def run(ctx: typer.Context):
    """Run code for namespace based on provided configurations"""
    name = ctx.obj.name
    name, name_dir = validate_and_create_name_folder(name)

    # If we made it to this point start to get the values from the config or from the CLI params
    # (cli will overwrite what already exists in the config)
    config_path = os.path.join(name_dir, CONFIG_FILE)
    # Try to read the config file which we will start to update with the params the user provided
    config_kwargs = bring_config_kwargs(config_path, name, name_dir)

    dir_path = ctx.obj.path or config_kwargs.get('path')
    if not valid_filepath(dir_path):
        typer.echo(f"invalid path {dir_path} - please update the config or provide a valid one in the CLI")
        raise typer.Exit(code=1)

    hardware = ctx.obj.hardware or config_kwargs.get('hardware')
    validate_hardware(hardware)

    # Grab any additional params the user provides
    cli_args = ctx.args
    user_params = build_context_params(cli_args) if cli_args else config_kwargs.get('params', {})

    # grab the docker related params - if none provided will have to build the dockerfile + image
    docker_client = docker.from_env()
    # which ever is not None (i.e. between the config, CLI, and what exists in the file system)
    dockerfile = get_path_to_dockerfile(dir_path, config_kwargs, ctx)

    # Try loading the local image if it exists
    path_to_image = f'{name}.tar'  # TODO update this path - maybe put in tmp folder?
    image_id = ctx.obj.image or name
    image = bring_image_from_docker_client(docker_client, image_id)

    # Create the image we need to run the code remotely
    build_and_save_image(image, path_to_image, dockerfile, docker_client, dir_path, name, name_dir,
                         user_params, hardware)

    # Now that we have all the updated args make sure the config file is updated for the next run
    typer.echo(f'[2/4] Updating config file for {name}')
    create_or_update_config_file(name_dir, name=name, hardware=hardware, user_params=user_params,
                                 dockerfile=dockerfile)

    typer.echo(f'[3/4] Running {name} with hardware {hardware}')

    # TODO Copy the image to remote server and run it there
    # run_image_on_remote_server(path_to_image, hardware=hardware)

    typer.echo(f'[4/4] Finished running {name}')
    raise typer.Exit()


@app.callback()
def common(ctx: typer.Context,
           dockerfile: str = typer.Option(None, '--dockerfile', '-d', help='path to existing dockerfile'),
           hardware: str = typer.Option(os.getenv('DEFAULT_HARDWARE'), '--hardware', '-h', help='desired hardware'),
           image: str = typer.Option(None, '--image', '-i', help='image id of existing local docker image'),
           name: str = typer.Option(None, '--name', '-n', help='name your microservice / URI'),
           path: str = typer.Option(None, '--path', '-p',
                                    help='Path to directory to be packaged and registered as a URI'),
           rename: Optional[str] = typer.Option(None, '--rename', '-r', callback=rename_callback,
                                                help='rename existing URI'),
           shell: bool = typer.Option(False, '--shell', '-s', help='run code in interactive mode'),
           version: Optional[bool] = typer.Option(None, '--version', '-v', callback=version_callback,
                                                  help='current package version')):
    """Welcome to Runhouse! Here's what you need to get started"""
    ctx.obj = Common(name, hardware, dockerfile, image, shell, path)
