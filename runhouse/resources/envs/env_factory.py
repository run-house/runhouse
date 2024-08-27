from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union

from runhouse.resources.packages import Package

from .conda_env import CondaEnv

from .env import Env
from .utils import _get_conda_yaml, _process_reqs


# generic Env factory method
def env(
    reqs: List[Union[str, Package]] = [],
    conda_env: Union[str, Dict] = None,
    name: Optional[str] = None,
    setup_cmds: List[str] = None,
    env_vars: Union[Dict, str] = {},
    working_dir: Optional[Union[str, Path]] = None,
    secrets: Optional[Union[str, "Secret"]] = [],
    compute: Optional[Dict] = {},
    load_from_den: bool = True,
    dryrun: bool = False,
):
    """Builds an instance of :class:`Env`.

    Args:
        reqs (List[str]): List of package names to install in this environment.
        conda_env (Union[str, Dict], optional): Dict representing conda env, Path to a conda env yaml file,
            or name of a local conda environment.
        name (Optional[str], optional): Name of the environment resource.
        setup_cmds (Optional[List[str]]): List of CLI commands to run for setup when the environment is
            being set up on a cluster.
        env_vars (Dict or str): Dictionary of environment variables, or relative path to .env file containing
            environment variables. (Default: {})
        working_dir (str or Path): Working directory of the environment, to be loaded onto the system.
            (Default: None)
        compute (Dict): Logical compute resources to be used by this environment, passed through to the
            cluster scheduler (generally Ray). Only use this if you know what you're doing.
            Example: ``{"cpus": 1, "gpus": 1}``. (Default: {})
            More info: https://docs.ray.io/en/latest/ray-core/scheduling/resources.html
        load_from_den (bool): Whether to try loading the Env resource from Den. (Default: ``True``)
        dryrun (bool, optional): Whether to run in dryrun mode. (Default: ``False``)


    Returns:
        Env: The resulting Env object.

    Example:
        >>> # regular python env
        >>> env = rh.env(reqs=["torch", "pip"])
        >>> env = rh.env(reqs=["reqs:./"], name="myenv")
        >>>
        >>> # conda env, see also rh.conda_env
        >>> conda_env_dict =
        >>>     {"name": "new-conda-env", "channels": ["defaults"], "dependencies": "pip", {"pip": "diffusers"})
        >>> conda_env = rh.env(conda_env=conda_env_dict)             # from a dict
        >>> conda_env = rh.env(conda_env="conda_env.yaml")           # from a yaml file
        >>> conda_env = rh.env(conda_env="local-conda-env-name")     # from a existing local conda env
        >>> conda_env = rh.env(conda_env="conda_env.yaml", reqs=["pip:/accelerate"])   # with additional reqs
    """
    if name and not any(
        [reqs, conda_env, setup_cmds, env_vars, secrets, working_dir, compute]
    ):
        try:
            return Env.from_name(name, load_from_den=load_from_den, dryrun=dryrun)
        except ValueError:
            return Env(name=name)

    if not name and compute:
        raise ValueError("Cannot specify compute to schedule an env on without a name.")

    reqs = _process_reqs(reqs or [])
    conda_yaml = _get_conda_yaml(conda_env)

    if conda_yaml:
        return CondaEnv(
            conda_yaml=conda_yaml,
            reqs=reqs,
            setup_cmds=setup_cmds,
            env_vars=env_vars,
            working_dir=working_dir,
            secrets=secrets,
            name=name or conda_yaml["name"],
            dryrun=dryrun,
        )

    return Env(
        reqs=reqs,
        setup_cmds=setup_cmds,
        env_vars=env_vars,
        working_dir=working_dir,
        secrets=secrets,
        name=name,
        compute=compute,
        dryrun=dryrun,
    )


# Conda Env factory method
def conda_env(
    reqs: List[Union[str, Package]] = [],
    conda_env: Union[str, Dict] = None,
    name: Optional[str] = None,
    setup_cmds: List[str] = None,
    env_vars: Optional[Dict] = {},
    working_dir: Optional[Union[str, Path]] = None,
    secrets: List[Union[str, "Secret"]] = [],
    compute: Optional[Dict] = {},
    dryrun: bool = False,
):
    """Builds an instance of :class:`CondaEnv`.

    Args:
        reqs (List[str]): List of package names to install in this environment.
        conda_env (Union[str, Dict], optional): Dict representing conda env, Path to a conda env yaml file,
            or name of a local conda environment.
        name (Optional[str], optional): Name of the environment resource.
        setup_cmds (Optional[List[str]]): List of CLI commands to run for setup when the environment is
            being set up on a cluster.
        env_vars (Dict or str): Dictionary of environment variables, or relative path to .env file containing
            environment variables. (Default: {})
        working_dir (str or Path): Working directory of the environment, to be loaded onto the system.
            (Default: None)
        compute (Dict): Logical compute resources to be used by this environment, passed through to the
            cluster scheduler (generally Ray). Only use this if you know what you're doing.
            Example: ``{"cpus": 1, "gpus": 1}``. (Default: {})
            More info: https://docs.ray.io/en/latest/ray-core/scheduling/resources.html
        dryrun (bool, optional): Whether to run in dryrun mode. (Default: ``False``)

    Returns:
        CondaEnv: The resulting CondaEnv object.

    Example:
        >>> rh.conda_env(reqs=["torch"])
        >>> rh.conda_env(reqs=["torch"], name="resource_name")
        >>> rh.conda_env(reqs=["torch"], name="resource_name", conda_env={"name": "conda_env"})
    """
    if not conda_env:
        if name:
            conda_env = {"name": name}
        else:
            conda_env = {"name": datetime.now().strftime("%Y%m%d_%H%M%S")}

    return env(
        reqs=reqs,
        conda_env=conda_env,
        name=name,
        setup_cmds=setup_cmds,
        env_vars=env_vars,
        working_dir=working_dir,
        secrets=secrets,
        compute=compute,
        dryrun=dryrun,
    )
