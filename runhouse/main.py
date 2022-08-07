"""This module provides the runhouse CLI."""
import warnings

# Suppress warnings when running commands
from cryptography.utils import CryptographyDeprecationWarning

warnings.filterwarnings("ignore", category=CryptographyDeprecationWarning)

import os
import logging
import typer
import time
import json
import subprocess
import pkg_resources
import webbrowser
from typing import Optional
import git
import validators as validators
from ray.dashboard.modules.job.sdk import JobSubmissionClient
from runhouse.common import Common
from runhouse.utils.utils import ERROR_FLAG, random_string_generator
from runhouse.config_parser import Config
from runhouse.utils.file_utils import create_directory, create_directories
from runhouse.utils.validation import validate_hardware, valid_filepath
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


def validate_and_get_name(ctx):
    name = ctx.obj.name
    if name is None:
        # first try to grab it from the latest env variable
        name = os.getenv('CURRENT_NAME')
        if name is None:
            # No name found, we will generate a random one for the user
            name = random_string_generator()
            typer.echo(f'No name found, using: {name}')
            dotenv.set_key(DOTENV_FILE, "CURRENT_NAME", name)
        else:
            typer.echo(f'Using name {name} (you can specify a new name with the --name command)')

    return name


def get_path_to_yaml_config_by_name(name):
    # TODO the yaml should be stored in runhouse server
    yaml_to_name = os.getenv('YAML_TO_NAME')
    if yaml_to_name is None:
        typer.echo(f'{ERROR_FLAG} No env variable found for "YAML_TO_NAME"')

    path_to_yaml = json.loads(yaml_to_name).get(name)

    return path_to_yaml


def update_env_vars_with_curr_name(name, ip_address=None):
    """keep current send in env variable to save user from constantly specifying with each command"""
    dotenv.set_key(DOTENV_FILE, "CURRENT_NAME", name)

    if ip_address is not None:
        curr_yaml = json.loads(os.getenv('YAML_TO_NAME', default='{}'))
        curr_yaml[name] = ip_address
        dotenv.set_key(DOTENV_FILE, "YAML_TO_NAME", json.dumps(curr_yaml))


def submit_job_to_ray_cluster(run_cmd, path_to_runnable_file, reqs_file, hardware_ip):
    try:
        client = JobSubmissionClient(f"http://{hardware_ip}:8265")
        job_id = client.submit_job(
            entrypoint=run_cmd,
            runtime_env={
                "working_dir": path_to_runnable_file,
                "pip": reqs_file
            }
        )
    except:
        typer.echo('Failed to run on ray cluster')
        raise typer.Exit(code=1)


def runnable_file_command(path_to_runnable_file, formatted_args):
    """Command to be stored in the .sh file in the rh subdirectory"""
    return f"{path_to_runnable_file} {formatted_args}"


def create_sh_file_in_dir(dir_name, file_name, text):
    with open(os.path.join(dir_name, file_name), 'w') as rsh:
        rsh.write(f'''#! /bin/sh\n{text}''')


def create_runnable_file_in_runhouse_subdir(name_dir, path_to_runnable_file_on_cluster, optional_cli_args):
    """Build the internal file in the runhouse directory used for executing the code the user wants to run remotely"""
    ext = os.path.splitext(path_to_runnable_file_on_cluster)[1]
    formatted_args: str = ' '.join(optional_cli_args)
    cmd = runnable_file_command(path_to_runnable_file_on_cluster, formatted_args)

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
    # TODO make ctx as optional args at the end (ex: runhouse send --name --hardware, etc.)
    start = time.time()

    name = validate_and_get_name(ctx)
    dotenv.set_key(DOTENV_FILE, "CURRENT_NAME", name)

    hardware = ctx.obj.hardware or os.getenv('DEFAULT_HARDWARE')
    # validate the hardware and grab its relevant IP address needed for building the send
    # validate_hardware(hardware)
    # hardware_ip = get_hostname_from_hardware(hardware)

    # update the env variables so we can access it
    # update_env_vars_with_curr_name(name, hardware_ip)

    typer.echo(f'[1/4] Starting to build send')

    package = ctx.obj.path or os.getcwd()
    if package.endswith('.git'):
        if not validators.url(package):
            typer.echo(
                f'{ERROR_FLAG} Invalid git url provided\nSample Url: https://github.com/<username>/<git-repo>.git')
            raise typer.Exit(code=1)

        full_path_to_package = internal_rh_name_dir
        # clone the package locally
        typer.echo(f'[2/4] Using github URL as package for the send ({package})')

        try:
            # TODO check auth for cloning private repos
            repo = git.Repo.clone_from(package)
        except git.GitCommandError as e:
            # clone either failed or already exists locally
            # TODO differentiate between failed clone vs. directory already exists error
            raise e
        full_path_to_package = repo.working_tree_dir
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

    Send(name=name,
         working_dir=full_path_to_package,
         reqs=reqs_file,
         cluster_ip=hardware_ip,
         )

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
    # TODO Note: need to first create the send - first check that name is in env var or they passed in name arg
    """Run a specific job on runhouse remote cluster"""
    start = time.time()
    # TODO we know where the send sits, now we just create the executable

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

    # Validate the path locally before creating the path to be used when running on the cluster
    runnable_file_name = ctx.obj.file or config_kwargs.get('file')
    if not runnable_file_name:
        # If we did not explicitly receive the path to the file (-f) by the user (and not provided in the config file)
        typer.echo(f'{ERROR_FLAG} Please include the name of to the file to run (using -f option)')
        raise typer.Exit(code=1)

    path_to_runnable_file = os.path.abspath(os.path.join(path_to_parent_dir, runnable_file_name))

    if not valid_filepath(path_to_runnable_file):
        # make sure the path the user provided is ok
        typer.echo(f'{ERROR_FLAG} No file found in path: {path_to_runnable_file}')
        raise typer.Exit(code=1)

    # TODO need to somehow know what the hash is (_ray_pkg_<hash of directory contents>)
    path_to_runnable_file_on_cluster = os.path.abspath(os.path.join(os.getenv('RAY_WORKING_DIR'),
                                                                    path_to_parent_dir,
                                                                    runnable_file_name))

    typer.echo(f'[3/5] Creating runhouse config for {name}')

    # Create a runnable file with the arguments provided in the internal runhouse subdirectory
    # The path to the file should match how we expect this to look on the server
    run_cmd: str = create_runnable_file_in_runhouse_subdir(internal_rh_name_dir, path_to_runnable_file_on_cluster,
                                                           optional_cli_args)

    reqs_file = find_requirements_file(path_to_runnable_file)
    if reqs_file is None:
        # have default requirements txt
        typer.echo('No requirements.txt found - will use default runhouse requirements')

    typer.echo(f'[4/5] Running {name} on remote cluster')

    submit_job_to_ray_cluster(run_cmd, path_to_runnable_file, reqs_file, hardware_ip)

    typer.echo(f'[5/5] Finished running {name} on remote cluster')

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