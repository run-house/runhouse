"""This module provides the runhouse CLI."""
import shutil
import warnings
from cryptography.utils import CryptographyDeprecationWarning

# Suppress warnings when running commands
warnings.filterwarnings("ignore", category=CryptographyDeprecationWarning)

import os
import json
import logging
import pkg_resources
from configparser import ConfigParser
from pathlib import Path
from typing import Optional
import docker
import typer
from runhouse.shell_handler import ShellHandler
from runhouse.ssh_manager import SSHManager
from runhouse.process_commands import process_cmd_commands
from runhouse.utils.utils import valid_filepath, random_string_generator, create_directory, delete_directory, \
    current_time
from runhouse.common import Common

# For now load from .env
from dotenv import load_dotenv
load_dotenv()

# # creates an explicit Typer application, app
app = typer.Typer(add_completion=False)

logger = logging.getLogger(__name__)

# create config object for reading / writing to runhouse config files
config = ConfigParser(strict=False, allow_no_value=True)

RUNHOUSE_DIR = os.path.join(os.getcwd(), Path(__file__).parent.parent.absolute(), 'runhouse')
MAIN_CONF_HEADER = 'main'
DOCKER_CONF_HEADER = 'docker'
RUNNABLE_FILE_NAME = 'run'
ERROR_FLAG = "[ERROR]"
MAX_DIR_LEN = 12

# map each hardware option to its IP / host
HARDWARE_TO_HOSTNAME = json.loads(os.getenv('HARDWARE_TO_HOSTNAME'))


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
    create_or_update_config_file(renamed_dir, name=new_dir_name, rename=True)


def validate_hardware(hardware):
    """Throw an error for invalid hardware specs"""
    if hardware not in list(HARDWARE_TO_HOSTNAME):
        typer.echo(f"{ERROR_FLAG} Invalid hardware specification")
        typer.echo(f"Please choose from the following options: {list(HARDWARE_TO_HOSTNAME)}")
        raise typer.Exit(code=1)


def get_hostname_from_hardware(hardware):
    """Based on mappings from hardware name to hostname IP"""
    hostname = HARDWARE_TO_HOSTNAME.get(hardware)

    if hostname is not None:
        return hostname

    typer.echo(f"{ERROR_FLAG} host name not found for hardware {hardware}")
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
        logger.error(f'{ERROR_FLAG} Unable to run image {path} on remote server: {e}')
        raise typer.Exit(code=1)


def create_directories_for_name(name_dir):
    # First make sure we have the parent runhouse dir
    create_directory(RUNHOUSE_DIR)

    # create the subdir of the named uri
    create_directory(name_dir)


def write_config(config_path):
    try:
        with open(config_path, 'w') as f:
            config.write(f)
    except:
        typer.echo(f'{ERROR_FLAG} Unable to save config file')
        raise typer.Exit(code=1)


def validate_name(name):
    # TODO maybe add some other checks on the directory name?
    if len(name) > MAX_DIR_LEN:
        typer.echo(f'{ERROR_FLAG} Runhouse does not support a name longer than ({MAX_DIR_LEN})')
        raise typer.Exit(code=1)


def create_or_update_config_file(directory, **kwargs):
    config_path = os.path.join(directory, os.getenv('CONFIG_FILE'))
    rename = kwargs.get('rename')

    if rename:
        if not valid_filepath(config_path):
            # If we are trying to rename an existing config make sure it still exists
            typer.echo(f'{ERROR_FLAG} Invalid path to config file')
            raise typer.Exit(code=1)

        # All we care about here is the actual "name" field defined in the config
        new_kwargs = read_config_file(config_path)
        new_kwargs['name'] = kwargs.get('name')
        new_kwargs['rename'] = False
        create_or_update_config_file(directory, **new_kwargs)
        return

    config.read(config_path)

    if not config.has_section(MAIN_CONF_HEADER):
        config.add_section(MAIN_CONF_HEADER)

    if not config.has_section(DOCKER_CONF_HEADER):
        config.add_section(DOCKER_CONF_HEADER)

    dockerfile = kwargs.get('dockerfile')
    rebuild = kwargs.get('rebuild')

    config.set(MAIN_CONF_HEADER, 'name', kwargs.get('name'))
    config.set(MAIN_CONF_HEADER, 'hardware', kwargs.get('hardware', os.getenv('DEFAULT_HARDWARE')))
    config.set(MAIN_CONF_HEADER, 'path', str(kwargs.get('path')))
    config.set(MAIN_CONF_HEADER, 'file', kwargs.get('file'))
    config.set(DOCKER_CONF_HEADER, 'dockerfile', dockerfile)
    config.set(DOCKER_CONF_HEADER, 'image_id', kwargs.get('image_id'))
    config.set(DOCKER_CONF_HEADER, 'image_path', kwargs.get('image_path'))

    if rebuild and dockerfile:
        # Only update the time added if the dockerfile was actually changed and it exists
        config.set(DOCKER_CONF_HEADER, 'time_added', str(current_time()))
    else:
        config.set(DOCKER_CONF_HEADER, 'time_added', None)

    write_config(config_path)


def read_config_file(config_path):
    config.read(config_path)

    # read values from file
    dockerfile = config.get(DOCKER_CONF_HEADER, 'dockerfile')
    image_id = config.get(DOCKER_CONF_HEADER, 'image_id')
    image_path = config.get(DOCKER_CONF_HEADER, 'image_path')
    dockerfile_timestamp = config.get(DOCKER_CONF_HEADER, 'time_added')

    name = config.get(MAIN_CONF_HEADER, 'name')
    hardware = config.get(MAIN_CONF_HEADER, 'hardware')
    path = config.get(MAIN_CONF_HEADER, 'path')
    file = config.get(MAIN_CONF_HEADER, 'file')

    return {'dockerfile': dockerfile, 'image_id': image_id, 'image_path': image_path, 'name': name,
            'hardware': hardware, 'path': path, 'file': file, 'time_added': dockerfile_timestamp}


def create_name_for_folder(name):
    if name is None:
        # if user did not provide a names we'll make one up
        name = random_string_generator().lower()
        typer.echo(f'Creating URI with name: {name}')
        return name

    # make sure the user provided name is in line with runhouse conventions
    validate_name(name)
    return name.lower()


def create_dockerfile(path_to_reqs, name_dir):
    # TODO make this cleaner
    text = f"""FROM {os.getenv('DOCKER_PYTHON_VERSION')}\nCOPY {path_to_reqs} 
    /opt/app/requirements.txt\nWORKDIR /opt/app\nRUN pip install -r {path_to_reqs}\nCOPY . .\nCMD [ "echo", "finished building image for {name_dir}" ]"""
    path_to_docker_file = os.path.join(name_dir, 'Dockerfile')
    with open(path_to_docker_file, 'w') as f:
        f.write(text)
    return path_to_docker_file


def build_and_save_image(image, path_to_image, path_to_reqs, dockerfile, docker_client, name, name_dir, hardware):
    success = True
    if not image:
        # if no image url has been provided we have some work to do
        # Need to build the image based on dockerfile provided, or if that isn't provided first build the dockerfile
        if not dockerfile:
            typer.echo('Building Dockerfile')
            dockerfile = create_dockerfile(path_to_reqs, name_dir)

        try:
            docker_client.images.build(path=name_dir, dockerfile=dockerfile, tag=name, labels={'hardware': hardware})
            typer.echo(f'[1/4] Successfully built image for {name}')
        except docker.errors.BuildError:
            typer.echo(f'{ERROR_FLAG} Failed to build image')
            success = False
        except docker.errors.APIError:
            typer.echo(f'{ERROR_FLAG} [API Error] Unable to build image')
            success = False

    else:
        # TODO this flow should change
        # if image exists then load into compressed format to be shipped off remotely
        save_image_to_tar(image, path_to_image)
        typer.echo(f"[1/4] Successfully loaded image {image.tags} with labels: {image.labels}")

    return success


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
        typer.echo(f'{ERROR_FLAG} Unable to retrieve image')
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


def get_path_to_dockerfile(path_to_parent_dir, config_kwargs, ctx):
    path_to_dockerfile = os.path.join(path_to_parent_dir, "Dockerfile")
    if valid_filepath(path_to_dockerfile):
        return path_to_dockerfile

    # if the dockerfile doesn't yet exist in filesystem try the user CLI params or the config file
    return ctx.obj.dockerfile or config_kwargs.get('dockerfile')


def build_path_to_parent_dir(ctx, config_kwargs):
    # Check if user has explicitly provided a path to the parent directory
    path_to_parent_dir = ctx.obj.path or config_kwargs.get('path')
    if path_to_parent_dir:
        return Path(path_to_parent_dir)

    # If the user gave us no indication of where the parent directory is assume its in the parent folder
    # (based on where the CLI command is being run)
    return Path(RUNHOUSE_DIR).parent.absolute()


def launch_local_docker_client():
    try:
        docker_client = docker.from_env()
        return docker_client
    except docker.errors.DockerException:
        typer.echo(f'{ERROR_FLAG} Docker client error')
        raise typer.Exit(code=1)


def default_image_name(name):
    return f'{name}_{int(current_time())}'


def bring_config_kwargs(config_path, name, file):
    if not valid_filepath(config_path):
        # If we don't have a config for this name yet define the initial default values
        return {'name': name, 'hardware': os.getenv('DEFAULT_HARDWARE'), 'file': file}

    # take from the config that already exists
    return read_config_file(config_path)


def create_runnable_file_in_runhouse_dir(path_to_runnable_file, name_dir, optional_cli_args):
    """Build the internal file in the runhouse directory used for executing the code the user wants to run remotely"""
    if path_to_runnable_file.endswith('.py'):
        runhouse_file = os.path.join(name_dir, f'{RUNNABLE_FILE_NAME}.py')
        shutil.copyfile(path_to_runnable_file, runhouse_file)
        if optional_cli_args:
            # TODO add the cli args the user gave as env variables in the run file
            pass
    elif path_to_runnable_file.endswith('.sh'):
        runhouse_file = os.path.join(name_dir, f'{RUNNABLE_FILE_NAME}.sh')
        shutil.copyfile(path_to_runnable_file, runhouse_file)
        if optional_cli_args:
            # TODO add the cli args the user gave as env variables in the run file
            pass
    else:
        # extension that we do not currently support
        typer.echo(f'{ERROR_FLAG} Please include a file with extension .py or .sh')
        raise typer.Exit(code=1)


def dockerfile_has_changed(time_added, path_to_dockerfile):
    """If the dockerfile has been updated since it was first created (as indicated in the config file)"""
    if not valid_filepath(path_to_dockerfile):
        typer.echo(f'{ERROR_FLAG} Unable to find dockerfile, please make sure it exists')
        raise typer.Exit(code=1)

    time_modified = os.path.getctime(path_to_dockerfile)

    return time_modified - time_added > 100


def user_rebuild_response(resp: str) -> bool:
    if resp.lower() not in ['yes', 'y', 'no', 'n']:
        typer.echo(f'{ERROR_FLAG} Invalid rebuild prompt')
        raise typer.Exit(code=1)
    return resp in ['yes', 'y']


def should_we_rebuild(provided_options, optional_args, config_path, config_kwargs, path_to_dockerfile) -> bool:
    """Determine if we need to rebuild the image based on the CLI arguments provided by the user + the config"""
    if not valid_filepath(config_path):
        # If there is no config in the name directory we definitely have to rebuild
        return True

    if optional_args:
        # If the user provided optional arguments to the runhouse command check whether they want to rebuild
        user_rebuild = typer.prompt('Optional args provided - would you like to rebuild? (Yes / Y or No / N)')
        if user_rebuild_response(user_rebuild):
            return True

    # If the user provided any options compare them to the the ones previously provided
    for arg_key, arg_val in provided_options.items():
        if arg_val != config_kwargs[arg_key]:
            # If any argument differs let's ask the user if they want to trigger a rebuild
            user_rebuild = typer.prompt('New option(s) provided - would you like to rebuild? (Yes / Y or No / N)')
            if user_rebuild_response(user_rebuild):
                return True

    dockerfile_time_added = config_kwargs.get('time_added')
    if dockerfile_time_added is None:
        # if no dockerfile exists we need to rebuild
        return True

    # check if the user changed the dockerfile manually (if so need to trigger a rebuild)
    rebuild = False
    if dockerfile_has_changed(float(dockerfile_time_added), path_to_dockerfile):
        user_rebuild = typer.prompt('Dockerfile has been updated - would you like to rebuild? (Yes / Y or No / N)')
        rebuild = user_rebuild_response(user_rebuild)

    return rebuild


@app.command(context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def run(ctx: typer.Context):
    """Run code for namespace based on provided configurations"""
    if ctx.obj.rename:
        # For renaming not running anything - just a directory change name and config changes
        rename_callback(ctx.obj.rename)
        raise typer.Exit()

    # If user chooses a one-time run we won't be saving anything new to the user's runhouse directory
    anon = ctx.obj.anon
    name = 'anon' if anon else create_name_for_folder(name=ctx.obj.name)

    # Directory for the name will sit in the runhouse folder
    name_dir = os.path.join(RUNHOUSE_DIR, name)
    config_path = os.path.join(name_dir, os.getenv('CONFIG_FILE'))

    path_to_runnable_file: str = ctx.obj.file
    # Try to read the config file which we will start to update with the params the user provided
    config_kwargs: dict = bring_config_kwargs(config_path, name, path_to_runnable_file)

    # Make sure hardware specs are valid
    hardware = ctx.obj.hardware or config_kwargs.get('hardware')
    validate_hardware(hardware)

    # Generate the path to where the dockerfile should live
    path_to_parent_dir = build_path_to_parent_dir(ctx, config_kwargs)
    dockerfile: str = get_path_to_dockerfile(name_dir, config_kwargs, ctx)

    # Check whether we need to rebuild the image or not
    optional_cli_args: list = ctx.args
    rebuild: bool = should_we_rebuild(provided_options=ctx.obj.user_provided_args, optional_args=optional_cli_args,
                                      config_path=config_path, config_kwargs=config_kwargs,
                                      path_to_dockerfile=dockerfile)

    # Path to requirements is assumed to be in the parent dir
    path_to_reqs = os.path.join(path_to_parent_dir, 'requirements.txt')
    if not valid_filepath(path_to_reqs):
        typer.echo(f'{ERROR_FLAG} No requirements.txt found in parent directory - please add before continuing')
        raise typer.Exit(code=1)

    # make sure we have the relevant directories created on the user's file system
    create_directories_for_name(name_dir)

    # Check if user provided the name of the file to be executed
    if not path_to_runnable_file:
        # If we did not explicitly receive the path to the file (-f) by the user (or not provided in the config file)
        typer.echo(f'{ERROR_FLAG} Please include the name of the file to run (using -f option)')
        raise typer.Exit(code=1)

    if not valid_filepath(path_to_runnable_file):
        # make sure the path the user provided is ok
        typer.echo(f'{ERROR_FLAG} No file found in path: {path_to_runnable_file}')
        raise typer.Exit(code=1)

    # Create a runnable file with the arguments provided in the internal runhouse directory
    create_runnable_file_in_runhouse_dir(path_to_runnable_file, name_dir, optional_cli_args)

    # Try loading the local image if it exists
    image_path = f'{name}.tar'  # TODO add specific path for image?
    # Give the image a default name if not given one (with current timestamp)
    image_id = ctx.obj.image or default_image_name(name)

    # If we were instructed to rebuild or the image still does not exist - build one from scratch
    if rebuild or image_id is None:
        typer.echo('Rebuilding the image')
        # grab the docker related params - if none provided will have to build the dockerfile + image
        docker_client = launch_local_docker_client()
        image = bring_image_from_docker_client(docker_client, image_id)

        # Create the image we need to run the code remotely
        image_exists = build_and_save_image(image, image_path, path_to_reqs, dockerfile, docker_client, name,
                                            name_dir, hardware)
        if image_exists:
            # If we succeeded in building the image and have it then let's run it
            typer.echo(f'[2/4] Running with hardware {hardware}')

            # TODO Copy the image to remote server and run it there - will prob be easier to change this up using k8s
            # run_image_on_remote_server(path_to_image, hardware=hardware)
            typer.echo(f'[3/4] Finished running')
        else:
            # If we failed to load the image then save as null in the config
            image_id = None
            image_path = None

    # even if we failed to build or failed to build image load the image still continue to set up the config
    # make sure the config file is updated for the next run
    typer.echo(f'[4/4] Updating config file')
    create_or_update_config_file(name_dir, path=path_to_parent_dir, file=path_to_runnable_file, name=name,
                                 hardware=hardware, dockerfile=dockerfile, image_id=image_id, image_path=image_path,
                                 rebuild=rebuild)

    if anon:
        # delete the directory which we needed to create for the one-time run
        delete_directory(name_dir)

    raise typer.Exit()


@app.callback()
def common(ctx: typer.Context,
           anon: bool = typer.Option(None, '--anon', '-a',
                                     help="anonymous/one-time run (run will not be named and config won't be created)"),
           dockerfile: str = typer.Option(None, '--dockerfile', '-d', help='path to existing dockerfile'),
           file: str = typer.Option(None, '--file', '-f',
                                    help='Specific file to run (ex: distributed_train.sh, bert_preprocessing.py'),
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
