"""This module provides the runhouse CLI."""
import warnings

# Suppress warnings when running commands
from cryptography.utils import CryptographyDeprecationWarning
warnings.filterwarnings("ignore", category=CryptographyDeprecationWarning)

import os
import logging
import typer
import glob
import time
import json
import subprocess
import pkg_resources
import webbrowser
from typing import Optional
from argparse import ArgumentParser
import git
import validators as validators
from ray.dashboard.modules.job.sdk import JobSubmissionClient
from runhouse.common import Common
from runhouse.utils.utils import ERROR_FLAG, random_string_generator
from runhouse.config_parser import Config
from runhouse.utils.file_utils import create_directory, create_directories
from runhouse.utils.validation import validate_hardware, valid_filepath, validate_runnable_file_path
from runhouse.rns.send import Send

logging.disable(logging.CRITICAL)

# create an explicit Typer application
app = typer.Typer(add_completion=False)

# Where to create the runhouse internal subdirectory (assume it should be in the user's current directory)
RUNHOUSE_DIR = os.path.join(os.getcwd(), 'rh')

# TODO We need to inject these in a different way (can't have the user have access to the variables)
import dotenv

DOTENV_FILE = dotenv.find_dotenv()
dotenv.load_dotenv(DOTENV_FILE)

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
    hardware_to_hostname = os.getenv('HARDWARE_TO_HOSTNAME')
    if hardware_to_hostname is None:
        typer.echo(f'{ERROR_FLAG} No env variable found for "HARDWARE_TO_HOSTNAME"')

    hostname = json.loads(hardware_to_hostname).get(hardware)

    if hostname is None:
        typer.echo(f"{ERROR_FLAG} host name not found for hardware {hardware}")
        raise typer.Exit(code=1)

    return hostname


def parse_cli_args(cli_args: list):
    """Parse the additional arguments the user provides to a given command"""
    # TODO this should be deprecated - these should be given as COMMANDS and not OPTIONS
    try:
        parser = ArgumentParser()
        parser.add_argument('--file', dest='file', help='file (with main) to be run on remote cluster', type=str)
        parser.add_argument('--hardware', dest='hardware', help='named hardware instance', type=str)
        parser.add_argument('--name', dest='name', help='name of the send', type=str)
        parser.add_argument('--package', dest='package', help='local directory or github URL', type=str)
        parser.add_argument('--path', dest='path', help='path to the file which will be run on remote cluster',
                            type=str)

        args = parser.parse_args(cli_args)
        return vars(args)

    except SystemExit:
        typer.echo(f'{ERROR_FLAG} Unable to parse options provided')
        raise typer.Exit(code=1)


def validate_and_get_name(ctx):
    name = ctx.obj.name
    if name is None:
        # first try to grab it from the latest env variable
        name = os.getenv('CURRENT_NAME')
        if name is None:
            # No name found, we will generate a random one for the user
            name = random_string_generator()
            typer.echo(f'No name found, using: {name}')
        else:
            typer.echo(f'Using name {name}, if you would like to create a new one specify with the --name command')

    return name


def get_path_to_yaml_config_by_name(name):
    # TODO the yaml should be stored in runhouse server
    yaml_to_name = os.getenv('YAML_TO_NAME')
    if yaml_to_name is None:
        typer.echo(f'{ERROR_FLAG} No env variable found for "YAML_TO_NAME"')

    path_to_yaml = json.loads(yaml_to_name).get(name)

    return path_to_yaml


def find_requirements_file(dir_path):
    return next(iter(glob.glob(f'{dir_path}/**/requirements.txt', recursive=True)), None)


def update_env_vars_with_curr_name(name, ip_address=None):
    """keep current send in env variable to save user from constantly specifying with each command"""
    dotenv.set_key(DOTENV_FILE, "CURRENT_NAME", name)

    if ip_address is not None:
        curr_yaml = json.loads(os.getenv('YAML_TO_NAME', {}))
        curr_yaml[name] = ip_address
        dotenv.set_key(DOTENV_FILE, "YAML_TO_NAME", json.dumps(curr_yaml))


def initialize_ray_cluster(full_path_to_package, reqs_file, hardware_ip, name):
    # try:
    Send(name=name,
         package_path=full_path_to_package,
         reqs=reqs_file,
         hardware=hardware_ip,
         )
    # except:
    #     typer.echo('Failed to deploy send to server')
    #     raise typer.Exit(code=1)


def runnable_file_command(path_to_runnable_file, formatted_args):
    """Command to be stored in the .sh file in the rh subdirectory"""
    return f"{path_to_runnable_file} {formatted_args}"


def create_sh_file_in_dir(dir_name, file_name, text):
    with open(os.path.join(dir_name, file_name), 'w') as rsh:
        rsh.write(f'''#! /bin/sh\n{text}''')


def create_runnable_file_in_runhouse_subdir(name_dir, path_to_runnable_file, optional_cli_args):
    """Build the internal file in the runhouse directory used for executing the code the user wants to run remotely"""
    ext = os.path.splitext(path_to_runnable_file)[1]
    formatted_args: str = ' '.join(optional_cli_args)
    cmd = runnable_file_command(path_to_runnable_file, formatted_args)

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


@app.command(context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def send(ctx: typer.Context):
    """Build serverless endpoint and deploy to remote cluster"""
    start = time.time()

    name = validate_and_get_name(ctx)

    # TODO move lots of the above into send constructor
    # Make sure we have the main rh directory in the local filesystem
    create_directory(RUNHOUSE_DIR)

    # validate the hardware and grab its relevant IP address needed for building the send
    hardware = ctx.obj.hardware or os.getenv('DEFAULT_HARDWARE')
    validate_hardware(hardware)
    hardware_ip = get_hostname_from_hardware(hardware)

    # update the env variables so we can access it
    update_env_vars_with_curr_name(name, hardware_ip)

    typer.echo(f'[1/4] Starting to build send')
    internal_rh_name_dir = os.path.join(RUNHOUSE_DIR, 'sends', name)
    create_directory(internal_rh_name_dir)

    package = ctx.obj.package
    if package is None:
        package = os.getcwd()

    if package.endswith('.git'):
        if not validators.url(package):
            typer.echo(
                f'{ERROR_FLAG} Invalid git url provided\nSample Url: https://github.com/<username>/<git-repo>.git')
            raise typer.Exit(code=1)

        full_path_to_package = internal_rh_name_dir
        # clone the package locally
        typer.echo(f'[2/4] Using github URL as package for the send ({package})')

        try:
            # TODO handle auth for cloning private repos
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
        typer.echo(f'{ERROR_FLAG} Package with path {full_path_to_package} not found')
        raise typer.Exit(code=1)

    reqs_file = find_requirements_file(full_path_to_package)
    if reqs_file is None:
        # have default requirements txt
        typer.echo('No requirements.txt found - will use default runhouse requirements')

    typer.echo(f'[3/4] Deploying send for {name}')

    initialize_ray_cluster(full_path_to_package, reqs_file, hardware_ip, name)

    typer.echo(f'[4/4] Finished deploying send, updating config')

    # create or update the config if it doesn't already exist
    # TODO figure out what parts of the config we want to keep (some can prob be deprecated)
    cfg.create_or_update_config_file(internal_rh_name_dir, path=full_path_to_package, file=None,
                                     name=name, hardware=hardware, dockerfile=None, image_tag=None,
                                     rebuild=None, config_kwargs={},
                                     container_root=None)

    end = time.time()
    typer.echo(f'Completed in {int(end - start)} seconds')

    raise typer.Exit()


@app.command(context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def ssh(ctx: typer.Context):
    """SSH into existing node on remote cluster"""
    name = validate_and_get_name(ctx)
    update_env_vars_with_curr_name(name)

    # use ray attach to ssh into the head node
    path_to_yaml = get_path_to_yaml_config_by_name(name)
    if path_to_yaml is None:
        typer.echo(f'{ERROR_FLAG} No yaml config found for {name}')
        raise typer.Exit(code=1)

    if not valid_filepath(path_to_yaml):
        typer.echo(f'{ERROR_FLAG} YAML file not found in path {path_to_yaml}')
        raise typer.Exit(code=1)

    try:
        subprocess.run(["ray", "attach", f"{path_to_yaml}"])
    except:
        typer.echo(f'{ERROR_FLAG} Unable to ssh into cluster')
        raise typer.Exit(code=1)

    raise typer.Exit()


@app.command(context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def run(ctx: typer.Context):
    """Run a specific job on runhouse remote cluster"""
    start = time.time()

    typer.echo(f'[1/5] Validating run params')
    optional_cli_args: list = ctx.args

    name = validate_and_get_name(ctx)
    update_env_vars_with_curr_name(name)

    # Directory for the name which will sit in the runhouse folder
    internal_rh_name_dir = os.path.join(RUNHOUSE_DIR, 'runs', name)
    config_path = os.path.join(internal_rh_name_dir, Config.CONFIG_FILE)

    # Try to read the config file which we will begin to populate with the params the user provided
    config_kwargs: dict = cfg.bring_config_kwargs(config_path, name)

    # Make sure hardware specs are valid
    hardware = ctx.obj.hardware or config_kwargs.get('hardware', os.getenv('DEFAULT_HARDWARE'))
    validate_hardware(hardware)
    hardware_ip = get_hostname_from_hardware(hardware)

    # Generate the path to the parent dir which will packaged for the run - if not specified use current working dir
    path_to_parent_dir = ctx.obj.path or config_kwargs.get('external_package', os.getcwd())

    # make sure we have the relevant directories created on the user's file system
    # create the main runhouse dir and the subdir of the named uri
    typer.echo(f'[2/5] Checking if we need to rebuild')
    create_directories(dir_names=[RUNHOUSE_DIR, internal_rh_name_dir])

    # Update the path to the runnable file with one defined in the config if it exists
    runnable_file_name = ctx.obj.file or config_kwargs.get('file')
    path_to_runnable_file = os.path.abspath(os.path.join(path_to_parent_dir, runnable_file_name))
    validate_runnable_file_path(path_to_runnable_file)

    typer.echo(f'[3/5] Creating runhouse config for {name}')

    # Create a runnable file with the arguments provided in the internal runhouse subdirectory
    run_cmd: str = create_runnable_file_in_runhouse_subdir(internal_rh_name_dir, path_to_runnable_file,
                                                           optional_cli_args)

    reqs_file = find_requirements_file(path_to_runnable_file)
    if reqs_file is None:
        # have default requirements txt
        typer.echo('No requirements.txt found - will use default runhouse requirements')

    typer.echo(f'[4/5] Running job for {name} on remote cluster')

    # TODO create the job with ray and submit it
    client = JobSubmissionClient(f"http://{hardware_ip}:8265")
    job_id = client.submit_job(
        entrypoint=run_cmd,
        runtime_env={
            "working_dir": path_to_runnable_file,
            "pip": reqs_file
        }
    )
    typer.echo(f'[5/5] Finished running on remote cluster')

    # TODO needs to be simplified - way too many fields here
    cfg.create_or_update_config_file(internal_rh_name_dir, path=path_to_parent_dir, file=path_to_runnable_file,
                                     name=name, hardware=hardware, dockerfile=None, image_tag=None,
                                     rebuild=None, config_kwargs=config_kwargs,
                                     container_root=None, package_tar=None)
    end = time.time()
    typer.echo(f'Run completed in {int(end - start)} seconds')

    raise typer.Exit()


@app.command(context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def jupyter(ctx: typer.Context):
    """Open a jupyter notebook instance on the runhouse remote cluster"""
    pass


@app.command(context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def login():
    """Login to the main runhouse web page"""
    try:
        webbrowser.open(f'{os.getenv("RUNHOUSE_WEBPAGE")}')
    except:
        typer.echo(f'{ERROR_FLAG} Failed to launch runhouse web page - please try again later')
        raise typer.Exit(code=1)


@app.callback()
def common(ctx: typer.Context,
           dockerfile: str = typer.Option(None, '--dockerfile', '-d', help='path to existing dockerfile'),
           file: str = typer.Option(None, '--file', '-f', help='path to specific file to run '
                                                               '(contained in the package provided'),
           hardware: str = typer.Option(None, '--hardware', '-h', help='desired hardware'),
           image: str = typer.Option(None, '--image', '-i', help='image tag of existing local docker image'),
           jupyter: bool = typer.Option(None, '--jupyter', '-j', help='open a jupyter notebook in runhouse cluster'),
           login: bool = typer.Option(None, '--login', '-l', help='login to the runhouse webpage'),
           name: str = typer.Option(None, '--name', '-n', help='name of the existing run or new run'),
           path: str = typer.Option(None, '--path', '-p', help='path to directory or github URL to be packaged '
                                                               'as part of the run'),
           ssh: bool = typer.Option(None, '--ssh', '-ssh', help='run code in shell on remote cluster'),
           send: bool = typer.Option(None, '--send', '-s',
                                     help='create a serverless endpoint from a local directory or git repo (.git)'),
           status: bool = typer.Option(None, '--status', '-status', help='check status of a send'),
           version: Optional[bool] = typer.Option(None, '--version', '-v', callback=version_callback,
                                                  help='current package version')):
    """Welcome to Runhouse! Here's what you need to get started"""
    # TODO some of these commands probably no longer needed
    ctx.obj = Common(name, hardware, dockerfile, file, image, ssh, path, send, status, login, jupyter)