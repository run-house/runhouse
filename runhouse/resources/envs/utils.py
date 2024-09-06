import logging
import subprocess

from pathlib import Path
from typing import Dict, List, Optional

import yaml

from runhouse.constants import CONDA_INSTALL_CMDS, EMPTY_DEFAULT_ENV_NAME
from runhouse.globals import rns_client
from runhouse.resources.resource import Resource
from runhouse.utils import locate_working_dir, run_with_logs


def _process_reqs(reqs):
    preprocessed_reqs = []
    for package in reqs:
        from runhouse.resources.packages import Package

        # TODO [DG] the following is wrong. RNS address doesn't have to start with '/'. However if we check if each
        #  string exists in RNS this will be incredibly slow, so leave it for now.
        if isinstance(package, str):
            if package[0] == "/" and rns_client.exists(package):
                # If package is an rns address
                package = rns_client.load_config(package)
            else:
                # if package refers to a local path package
                path = Path(Package.split_req_install_method(package)[1]).expanduser()
                if path.is_absolute() or (locate_working_dir() / path).exists():
                    package = Package.from_string(package)
        elif isinstance(package, dict):
            package = Package.from_config(package)
        preprocessed_reqs.append(package)
    return preprocessed_reqs


def _process_env_vars(env_vars):
    processed_vars = (
        _env_vars_from_file(env_vars) if isinstance(env_vars, str) else env_vars
    )
    return processed_vars


def _get_env_from(env, load: bool = True):
    if isinstance(env, Resource):
        return env

    from runhouse.resources.envs import Env

    if isinstance(env, List):
        if len(env) == 0:
            return Env(reqs=env, working_dir=None)
        return Env(reqs=env)
    elif isinstance(env, Dict):
        return Env.from_config(env)
    elif isinstance(env, str) and EMPTY_DEFAULT_ENV_NAME not in env:
        if not load:
            return env

        try:
            return (
                Env.from_name(env)
                if rns_client.exists(env, resource_type="env")
                else env
            )
        except ValueError:
            return env
    return env


def _get_conda_yaml(conda_env=None):
    if not conda_env:
        return None
    if isinstance(conda_env, str):
        if Path(conda_env).expanduser().exists():  # local yaml path
            conda_yaml = yaml.safe_load(open(conda_env))
        elif f"\n{conda_env} " in subprocess.check_output(
            "conda info --envs".split(" ")
        ).decode("utf-8"):
            res = subprocess.check_output(
                f"conda env export -n {conda_env} --no-build".split(" ")
            ).decode("utf-8")
            conda_yaml = yaml.safe_load(res)
        else:
            raise Exception(
                f"{conda_env} must be a Dict or point to an existing path or conda environment."
            )
    else:
        conda_yaml = conda_env

    # ensure correct version to Ray -- this is subject to change if SkyPilot adds additional ray version support
    conda_yaml["dependencies"] = (
        conda_yaml["dependencies"] if "dependencies" in conda_yaml else []
    )
    if not [dep for dep in conda_yaml["dependencies"] if "pip" in dep]:
        conda_yaml["dependencies"].append("pip")
    if not [
        dep
        for dep in conda_yaml["dependencies"]
        if isinstance(dep, Dict) and "pip" in dep
    ]:
        conda_yaml["dependencies"].append({"pip": ["ray >= 2.2.0, != 2.6.0"]})
    else:
        for dep in conda_yaml["dependencies"]:
            if (
                isinstance(dep, Dict)
                and "pip" in dep
                and not [pip for pip in dep["pip"] if "ray" in pip]
            ):
                dep["pip"].append("ray >= 2.2.0, != 2.6.0")
                continue
    return conda_yaml


def _env_vars_from_file(env_file):
    try:
        from dotenv import dotenv_values, find_dotenv
    except ImportError:
        raise ImportError(
            "`dotenv` package is needed. You can install it with `pip install python-dotenv`."
        )

    dotenv_path = find_dotenv(str(env_file), usecwd=True)
    env_vars = dotenv_values(dotenv_path)
    return dict(env_vars)


# ------- Installation helpers -------
def run_setup_command(
    cmd: str,
    cluster: "Cluster" = None,
    env_vars: Dict = None,
    stream_logs: bool = True,
    node: Optional[str] = None,
):
    """
    Helper function to run a command during possibly the cluster default env setup. If a cluster is provided,
    run command on the cluster using SSH. If the cluster is not provided, run locally, as if already on the
    cluster (rpc call).

    Args:
        cmd (str): Command to run on the
        cluster (Optional[Cluster]): (default: None)
        stream_logs (bool): (default: True)

    Returns:
       (status code, stdout)
    """
    if not cluster:
        return run_with_logs(cmd, stream_logs=stream_logs, require_outputs=True)[:2]
    elif cluster.on_this_cluster():
        cmd = cluster.default_env._full_command(cmd)
        return run_with_logs(cmd, stream_logs=stream_logs, require_outputs=True)[:2]

    return cluster._run_commands_with_runner(
        [cmd], stream_logs=stream_logs, env_vars=env_vars, node=node
    )[0]


def install_conda(cluster: "Cluster" = None, node: Optional[str] = None):
    if run_setup_command("conda --version", cluster=cluster, node=node)[0] != 0:
        logging.info("Conda is not installed. Installing...")
        for cmd in CONDA_INSTALL_CMDS:
            run_setup_command(cmd, cluster=cluster, node=node, stream_logs=True)
        if run_setup_command("conda --version", cluster=cluster, node=node)[0] != 0:
            raise RuntimeError("Could not install Conda.")
