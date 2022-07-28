"""This module provides the runhouse CLI."""
import warnings

# Suppress warnings when running commands
from cryptography.utils import CryptographyDeprecationWarning

warnings.filterwarnings("ignore", category=CryptographyDeprecationWarning)

import os
import typer
import glob
import time
import json
import pkg_resources
from typing import Optional
from argparse import ArgumentParser
import ray
import git
from runhouse.common import Common
from runhouse.utils.utils import ERROR_FLAG
from runhouse.config_parser import Config
from runhouse.utils.file_utils import create_directory
from runhouse.utils.validation import validate_hardware, valid_filepath

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


def find_requirements_file(dir_path):
    return next(iter(glob.glob(f'{dir_path}/**/requirements.txt', recursive=True)), None)


def initialize_ray_cluster(full_path_to_package, reqs_file, hardware_ip, name):
    try:
        runtime_env = {"working_dir": full_path_to_package,
                       "pip": reqs_file,
                       'env_vars': dict(os.environ),
                       'excludes': ['*.log', '*.tar', '*.tar.gz', '.env', 'venv', '.idea', '.DS_Store', '__pycache__',
                                    '*.whl']}
        # use the remote cluster head node's IP address
        ray.init(f'ray://{hardware_ip}:10001',
                 namespace=name,
                 runtime_env=runtime_env)
    except:
        typer.echo('Failed to deploy send to server')
        raise typer.Exit(code=1)


@app.command(context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def run(ctx: typer.Context):
    # TODO separaate these into separate functions (send, ssh, etc.)
    """Run code for given name based on provided configurations"""
    start = time.time()

    optional_cli_args: list = ctx.args
    parsed_cli_args: dict = parse_cli_args(optional_cli_args)

    send = ctx.obj.send
    ssh = ctx.obj.ssh

    name = ctx.obj.name or parsed_cli_args.get('name')
    if name is None:
        typer.echo('No name provided for the run')
        raise typer.Exit(code=1)

    # Make sure we have the main rh directory in the local filesystem
    create_directory(RUNHOUSE_DIR)

    # validate the hardware and grab its relevant IP address needed for building the send
    hardware = ctx.obj.hardware or os.getenv('DEFAULT_HARDWARE')
    validate_hardware(hardware)
    hardware_ip = get_hostname_from_hardware(hardware)

    if send:
        typer.echo(f'[1/4] Starting to build send package')
        internal_rh_name_dir = os.path.join(RUNHOUSE_DIR, 'sends', name)
        create_directory(internal_rh_name_dir)

        package = parsed_cli_args.get('package')
        config_path, config_kwargs = bring_config_path_and_kwargs(internal_rh_name_dir, name)

        if package.endswith('.git'):
            full_path_to_package = internal_rh_name_dir
            # clone the package locally
            typer.echo(f'[2/4] Using github URL as package for the send ({package})')
            try:
                git.Git(internal_rh_name_dir).clone(package)
            except git.GitCommandError:
                # clone either failed or already exists locally
                # TODO differentiate between failed clone vs. directory already exists error
                pass
        else:
            # package refers to local directory
            typer.echo(f'[2/4] Using local directory to be packaged for the send ({package})')
            full_path_to_package = os.path.abspath(package)

        # Using a local directory as a package
        if not valid_filepath(full_path_to_package):
            typer.echo(f'Package with path {full_path_to_package} not found')
            raise typer.Exit(code=1)

        reqs_file = find_requirements_file(full_path_to_package)
        if reqs_file is None:
            # have default requirements txt
            typer.echo('No requirements.txt found - will use default runhouse requirements')

        initialize_ray_cluster(full_path_to_package, reqs_file, hardware_ip, name)

        typer.echo(f'[3/4] Finished building and deploying send for {name}')

    elif ssh:
        # use ray attach to ssh into the head node
        # TODO the yaml should be stored in runhouse server
        path_to_yaml = os.getenv('PATH_TO_RAY_YAML')
        try:
            os.system(f"ray attach {path_to_yaml}")
        except:
            typer.echo(f'{ERROR_FLAG} Unable to ssh into cluster')
            raise typer.Exit(code=1)
        raise typer.Exit()

    else:
        typer.echo(f'No valid command provided')
        raise typer.Exit(code=1)

    typer.echo(f'[4/4] Finished running, updating config')

    # create or update the config if it doesn't already exist
    cfg.create_or_update_config_file(internal_rh_name_dir, path=full_path_to_package, file=None,
                                     name=name, hardware=hardware, dockerfile=None, image_tag=None,
                                     rebuild=None, config_kwargs=config_kwargs,
                                     container_root=None)

    end = time.time()
    typer.echo(f'Completed in {int(end - start)} seconds')

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
           status: bool = typer.Option(None, '--status', '-status', help='check status of a send'),
           version: Optional[bool] = typer.Option(None, '--version', '-v', callback=version_callback,
                                                  help='current package version')):
    """Welcome to Runhouse! Here's what you need to get started"""
    ctx.obj = Common(name, hardware, dockerfile, file, image, ssh, path, send, status)
