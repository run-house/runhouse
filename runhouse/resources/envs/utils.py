import subprocess

from pathlib import Path
from typing import Dict, List

import yaml

from runhouse.constants import EMPTY_DEFAULT_ENV_NAME
from runhouse.globals import rns_client
from runhouse.resources.resource import Resource
from runhouse.utils import locate_working_dir


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
