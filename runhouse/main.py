"""This module provides the runhouse CLI."""
import warnings

# Suppress warnings when running commands
import pkg_resources
from cryptography.utils import CryptographyDeprecationWarning

warnings.filterwarnings("ignore", category=CryptographyDeprecationWarning)

import os
import json
from pathlib import Path
import typer
from typing import Optional
from runhouse.common import Common
from runhouse.shell_handler import ShellHandler
from runhouse.ssh_manager import SSHManager
from runhouse.process_commands import process_cmd_commands
from runhouse.utils.utils import ERROR_FLAG
from runhouse.config import Config
from runhouse.utils.docker_utils import dockerfile_has_changed, get_path_to_dockerfile, default_image_name, \
    launch_local_docker_client, bring_image_from_docker_client, build_and_save_image
from runhouse.utils.file_utils import copy_runnable_file_to_runhouse_subdir, create_name_for_folder, delete_directory, \
    create_directory
from runhouse.utils.validation import validate_runnable_file_path, validate_name, validate_hardware, valid_filepath

# For now load from .env
from dotenv import load_dotenv

load_dotenv()

# # creates an explicit Typer application, app
app = typer.Typer(add_completion=False)

RUNHOUSE_DIR = os.path.join(os.getcwd(), Path(__file__).parent.parent.absolute(), 'runhouse')

# map each hardware option to its IP / host
HARDWARE_TO_HOSTNAME = json.loads(os.getenv('HARDWARE_TO_HOSTNAME'))

# Create config object used for managing file actions
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
        typer.echo(f'{ERROR_FLAG} Unable to run image {path} on remote server: {e}')
        raise typer.Exit(code=1)


def build_path_to_parent_dir(ctx, config_kwargs):
    # Check if user has explicitly provided a path to the parent directory
    path_to_parent_dir = ctx.obj.path or config_kwargs.get('path')
    if path_to_parent_dir:
        return Path(path_to_parent_dir)

    # If the user gave us no indication of where the parent directory is assume its in the parent folder
    # (based on where the CLI command is being run)
    return Path(RUNHOUSE_DIR).parent.absolute()


def create_runnable_file_in_runhouse_subdir(path_to_runnable_file, name_dir, optional_cli_args):
    """Build the internal file in the runhouse directory used for executing the code the user wants to run remotely"""
    ext = os.path.splitext(path_to_runnable_file)[1]
    if ext not in ['.py', '.sh']:
        typer.echo(f'{ERROR_FLAG} Runhouse currently supports file types with extensions .py or .sh')
        raise typer.Exit(code=1)
    elif ext == '.py':
        copy_runnable_file_to_runhouse_subdir(path_to_runnable_file, name_dir, ext)
        if optional_cli_args:
            # TODO can't think of an elegant way to add the args as env variables to this run.py file
            pass
    else:
        copy_runnable_file_to_runhouse_subdir(path_to_runnable_file, name_dir, ext)
        if optional_cli_args:
            # TODO add the cli args the user gave as env variables in the run file
            pass


def user_rebuild_response(resp: str) -> bool:
    if resp.lower() not in ['yes', 'y', 'no', 'n']:
        typer.echo(f'{ERROR_FLAG} Invalid rebuild prompt')
        raise typer.Exit(code=1)
    return resp in ['yes', 'y']


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

    path_to_runnable_file: str = ctx.obj.file
    # Try to read the config file which we will begin to populate with the params the user provided
    config_kwargs: dict = cfg.bring_config_kwargs(config_path, name, path_to_runnable_file)

    # Make sure hardware specs are valid
    hardware = ctx.obj.hardware or config_kwargs.get('hardware')
    validate_hardware(hardware, hardware_to_hostname=HARDWARE_TO_HOSTNAME)

    # Generate the path to where the dockerfile should live
    path_to_parent_dir = build_path_to_parent_dir(ctx, config_kwargs)
    dockerfile: str = get_path_to_dockerfile(name_dir, config_kwargs, ctx)

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

    # make sure we have the relevant directories (parent + named subdir) created on the user's file system
    create_directory(RUNHOUSE_DIR)
    # create the subdir of the named uri
    create_directory(name_dir)

    # Check if user provided the name of the file to be executed
    validate_runnable_file_path(path_to_runnable_file)

    # Create a runnable file with the arguments provided in the internal runhouse subdirectory
    create_runnable_file_in_runhouse_subdir(path_to_runnable_file, name_dir, optional_cli_args)

    # Try loading the local image if it exists
    image_path = f'{name}.tar'  # TODO add specific folder for saving images (or maybe in /tmp)?
    image_id = ctx.obj.image

    # If we were instructed to rebuild or the image still does not exist - build one from scratch
    if rebuild or image_id is None:
        typer.echo('Rebuilding the image')
        # Give the image a default name if not given one (with current timestamp)
        image_id = default_image_name(name)
        # grab the docker related params - if none provided will have to build the dockerfile + image
        docker_client = launch_local_docker_client()
        image = bring_image_from_docker_client(docker_client, image_id)

        # Create the image we need to run the code remotely
        image_exists = build_and_save_image(image, image_path, path_to_reqs, dockerfile, docker_client, name,
                                            name_dir, hardware)
        if image_exists:
            # If we succeeded in building the image and have it then let's run it
            typer.echo(f'[2/3] Running with hardware {hardware}')

            # TODO Copy the image to remote server and run it there - will prob be easier to change this up using k8s
            # run_image_on_remote_server(path_to_image, hardware=hardware)
            typer.echo(f'[3/3] Finished running')
        else:
            # If we failed to load the image then save as null in the config
            image_id = None
            image_path = None

    # even if we failed to build or failed to build image load the image still continue to set up the config
    # make sure the config file is updated for the next run
    cfg.create_or_update_config_file(name_dir, path=path_to_parent_dir, file=path_to_runnable_file, name=name,
                                     hardware=hardware, dockerfile=dockerfile, image_id=image_id, image_path=image_path,
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
