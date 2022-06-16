"""This module provides the runhouse CLI."""
import warnings
from cryptography.utils import CryptographyDeprecationWarning

# Suppress warnings when running commands
warnings.filterwarnings("ignore", category=CryptographyDeprecationWarning)

import os
import json
import logging
import pkg_resources
from dataclasses import dataclass
from configparser import ConfigParser
from pathlib import Path
from typing import Optional
import docker
import typer
from runhouse.shell_handler import ShellHandler
from runhouse.ssh_manager import SSHManager
from runhouse.process_commands import process_cmd_commands
from runhouse.utils.utils import valid_filepath, random_string_generator, create_directory, delete_directory, error_flag

# For now load from .env
from dotenv import load_dotenv
load_dotenv()

# # creates an explicit Typer application, app
app = typer.Typer(add_completion=False)

logger = logging.getLogger(__name__)

# create config object for reading / writing to runhouse config files
config = ConfigParser()

# TODO maybe make this path configurable?
RUNHOUSE_DIR = os.path.join(os.getcwd(), Path(__file__).parent.parent.absolute(), 'runhouse')

CONFIG_FILE = 'config.ini'
MAIN_CONF_HEADER = 'main'
DOCKER_CONF_HEADER = 'docker'
# map each hardware option to its IP / host
HARDWARE_TO_HOSTNAME = json.loads(os.getenv('HARDWARE_TO_HOSTNAME'))


def version_callback(value: bool) -> None:
    if value:
        name = 'runhouse'
        version = pkg_resources.get_distribution(name).version
        typer.echo(f"{name}=={version}")
        raise typer.Exit()


def rename_callback(name: str) -> None:
    if name is None:
        return None

    name_dir = os.path.join(RUNHOUSE_DIR, name)
    if not valid_filepath(name_dir):
        typer.echo(f'{error_flag()} {name} not found in {RUNHOUSE_DIR}')
        return None

    # TODO rather than a prompt allow the user to specify in the same command (similar to mv <src> <dest>)
    new_dir_name = typer.prompt("Please enter the new name")
    renamed_dir = os.path.join(RUNHOUSE_DIR, new_dir_name)
    if any([name for name in os.listdir(RUNHOUSE_DIR) if name == new_dir_name]):
        typer.echo(f'{error_flag()} {new_dir_name} already exists, cannot rename')
        return None
    os.rename(name_dir, renamed_dir)
    create_or_update_config_file(renamed_dir, name=new_dir_name, update=True)


def validate_hardware(hardware):
    """Throw an error for invalid hardware specs"""
    if hardware not in list(HARDWARE_TO_HOSTNAME):
        typer.echo(f"{error_flag()} Invalid hardware specification")
        typer.echo(f"Please choose from the following options: {list(HARDWARE_TO_HOSTNAME)}")
        raise typer.Exit(code=1)


def get_hostname_from_hardware(hardware):
    """Based on mappings from hardware name to hostname IP"""
    hostname = HARDWARE_TO_HOSTNAME.get(hardware)

    if hostname is not None:
        return hostname

    typer.echo(f"{error_flag()} host name not found for hardware {hardware}")
    typer.Exit(code=1)


def open_bash_on_remote_server(hardware):
    """Open bash on remote env in user """
    host = get_hostname_from_hardware(hardware)
    typer.echo(f"Opening shell with {hardware} on host {host}\n")
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
        logger.error(f'{error_flag()} Unable to run image {path} on remote server: {e}')
        raise typer.Exit(code=1)


def create_directories_for_name(name_dir):
    # First make sure we have the parent "runhouse" dir
    create_directory(RUNHOUSE_DIR)

    # create the subdir of the named uri
    create_directory(name_dir)


def create_or_update_config_file(directory, **kwargs):
    config_path = os.path.join(directory, CONFIG_FILE)
    if kwargs.get('update') and not valid_filepath(config_path):
        # If we are trying to update an existing config make sure it still exists
        typer.echo(f'{error_flag()} Invalid path to config file')
        raise typer.Exit(code=1)

    config.read('config.ini')
    config.add_section(MAIN_CONF_HEADER)
    config.add_section(DOCKER_CONF_HEADER)

    name = kwargs.get('name')
    hardware = kwargs.get('hardware', os.getenv('DEFAULT_HARDWARE'))
    dockerfile = kwargs.get('dockerfile')

    config.set(MAIN_CONF_HEADER, 'name', name)
    config.set(MAIN_CONF_HEADER, 'hardware', hardware)
    config.set(MAIN_CONF_HEADER, 'path', directory)
    config.set(DOCKER_CONF_HEADER, 'dockerfile', dockerfile)

    try:
        with open(config_path, 'w') as f:
            config.write(f)
    except:
        typer.echo(f'{error_flag()} Unable to save config file')
        raise typer.Exit(code=1)


def read_config_file(config_path):
    config.read(config_path)

    # read values from file
    dockerfile = config.get(DOCKER_CONF_HEADER, 'dockerfile')
    name = config.get(MAIN_CONF_HEADER, 'name')
    hardware = config.get(MAIN_CONF_HEADER, 'hardware')
    path = config.get(MAIN_CONF_HEADER, 'path')

    return {'dockerfile': dockerfile, 'name': name, 'hardware': hardware, 'path': path}


def validate_and_create_name_folder(name):
    if name is None:
        # if user did not provide a namespace for this run we'll make one up
        name = random_string_generator().lower()
        typer.echo(f'Creating new runhouse URI with name: {name}')
    else:
        # For the purposes of building images (and using it as a tag), make sure it is always lower
        name = name.lower()

    name_dir = os.path.join(RUNHOUSE_DIR, name)
    # make sure we have the relevant directories created on the user's file system
    create_directories_for_name(name_dir)

    return name, name_dir


def create_dockerfile(path_to_reqs, name_dir):
    # TODO make this cleaner
    text = f"""FROM {os.getenv('DOCKER_PYTHON_VERSION')}\nCOPY {path_to_reqs} 
    /opt/app/requirements.txt\nWORKDIR /opt/app\nRUN pip install -r {path_to_reqs}\nCOPY . .\nCMD [ "python", "{name_dir}" ]"""
    path_to_docker_file = os.path.join(name_dir, 'Dockerfile')
    with open(path_to_docker_file, 'w') as f:
        f.write(text)
    return path_to_docker_file


def build_and_save_image(image, path_to_image, dockerfile, docker_client, name, name_dir, hardware):
    if not image:
        # if no image url has been provided we have some work to do
        # Need to build the image based on dockerfile provided, or if that isn't provided first build the dockerfile
        if not dockerfile:
            # Check for requirements
            path_to_reqs = os.path.join(Path(RUNHOUSE_DIR).parent.absolute(), "requirements.txt")
            if not valid_filepath(path_to_reqs):
                typer.echo(f'{error_flag()} No requirements.txt found in root directory, please add before continuing')
                raise typer.Exit(code=1)
            else:
                # TODO create the dockerfile
                typer.echo('Building Dockerfile with provided requirements')
                dockerfile = create_dockerfile(path_to_reqs, name_dir)

        typer.echo(f'[1/4] Building image for {name}')

        try:
            docker_client.images.build(path=name_dir, dockerfile=dockerfile, tag=name, labels={'hardware': hardware})
        except docker.errors.BuildError:
            typer.echo(f'{error_flag()} Failed to build image - check the path')
            raise typer.Exit(code=1)
        except docker.errors.APIError:
            typer.echo(f'{error_flag()} Unable to build docker image')
            raise typer.Exit(code=1)

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
        typer.echo(f'{error_flag()} Error with docker client retrieving image')
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


@dataclass
class Common:
    name: str
    hardware: str
    dockerfile: str
    image: str
    shell: bool
    path: str
    anon: bool
    rename: str


@app.command(context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def run(ctx: typer.Context):
    """Run code for namespace based on provided configurations"""
    if ctx.obj.rename:
        rename_callback(ctx.obj.rename)
        raise typer.Exit()

    anon = ctx.obj.anon
    if anon:
        # If user chooses a one-time run we won't be saving anything new to the user's runhouse directory
        config_kwargs = {}
        name_dir = ctx.obj.path
        name = 'anom'
        if name_dir is None:
            # TODO or we just take the current directory and assume that's what the user wants?
            typer.echo(f'{error_flag()} For an anonymous run please provide a path param to the '
                       f'relevant directory (use -p)')
            raise typer.Exit(code=1)
    else:
        name, name_dir = validate_and_create_name_folder(name=ctx.obj.name)
        # If we made it to this point start to get the values from the config or from the CLI params
        # (cli will overwrite what already exists in the config)
        config_path = os.path.join(name_dir, CONFIG_FILE)
        # Try to read the config file which we will start to update with the params the user provided
        config_kwargs = bring_config_kwargs(config_path, name, name_dir)

    dir_path = ctx.obj.path or config_kwargs.get('path', "")
    if not valid_filepath(dir_path):
        typer.echo(f"{error_flag()} Invalid path or no path provided - please update the config or provide a valid one "
                   f"in the CLI")
        raise typer.Exit(code=1)

    hardware = ctx.obj.hardware or config_kwargs.get('hardware')
    validate_hardware(hardware)

    # TODO - so don't store this? figure out what to do with these args
    cli_args = ctx.args

    # grab the docker related params - if none provided will have to build the dockerfile + image
    docker_client = docker.from_env()
    # which ever is not None (i.e. between the config, CLI, and what exists in the file system)
    dockerfile = get_path_to_dockerfile(dir_path, config_kwargs, ctx)

    # Try loading the local image if it exists
    # TODO update this path - maybe put in tmp folder?
    path_to_image = f'{name}.tar'
    image_id = ctx.obj.image or name
    image = bring_image_from_docker_client(docker_client, image_id)

    # Create the image we need to run the code remotely
    build_and_save_image(image, path_to_image, dockerfile, docker_client, name, name_dir, hardware)

    # Now that we have all the updated args make sure the config file is updated for the next run
    typer.echo(f'[2/4] Updating config file for {name}')
    create_or_update_config_file(name_dir, name=name, hardware=hardware, dockerfile=dockerfile)

    typer.echo(f'[3/4] Running {name} with hardware {hardware}')

    # TODO Copy the image to remote server and run it there - will prob be easier to change this up using k8s
    # run_image_on_remote_server(path_to_image, hardware=hardware)

    typer.echo(f'[4/4] Finished running {name}')
    if anon:
        # delete the directory which we needed to create for the one-time run
        delete_directory(name_dir)

    raise typer.Exit()


@app.callback()
def common(ctx: typer.Context,
           anon: bool = typer.Option(False, '--anon', '-a',
                                     help='anonymous/one-time run (no config or metadata saved)'),
           dockerfile: str = typer.Option(None, '--dockerfile', '-d', help='path to existing dockerfile'),
           hardware: str = typer.Option(os.getenv('DEFAULT_HARDWARE'), '--hardware', '-h', help='desired hardware'),
           image: str = typer.Option(None, '--image', '-i', help='image id of existing local docker image'),
           name: str = typer.Option(None, '--name', '-n', help='name your microservice / URI'),
           path: str = typer.Option(None, '--path', '-p',
                                    help='Path to parent directory to be packaged and registered as a URI'),
           shell: bool = typer.Option(False, '--shell', '-s', help='run code in interactive mode'),
           rename: str = typer.Option(None, '--rename', '-r', help='rename existing URI'),
           version: Optional[bool] = typer.Option(None, '--version', '-v', callback=version_callback,
                                                  help='current package version')):
    """Welcome to Runhouse! Here's what you need to get started"""
    ctx.obj = Common(name, hardware, dockerfile, image, shell, path, anon, rename)
