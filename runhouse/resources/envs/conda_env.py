import subprocess

from pathlib import Path
from typing import Dict, List, Optional, Union

import yaml

from runhouse.constants import CONDA_PREFERRED_PYTHON_VERSION, ENVS_DIR
from runhouse.globals import obj_store
from runhouse.logger import get_logger

from runhouse.resources.envs.utils import install_conda, run_setup_command
from runhouse.resources.packages import Package

from .env import Env

logger = get_logger(__name__)


class CondaEnv(Env):
    RESOURCE_TYPE = "env"

    def __init__(
        self,
        conda_yaml: Union[str, Dict],
        name: Optional[str] = None,
        reqs: List[Union[str, Package]] = [],
        setup_cmds: List[str] = None,
        env_vars: Optional[Dict] = {},
        working_dir: Optional[Union[str, Path]] = "./",
        secrets: List[Union[str, "Secret"]] = [],
        dryrun: bool = True,
        **kwargs,  # We have this here to ignore extra arguments when calling from_config
    ):
        """
        Runhouse CondaEnv object.

        .. note::
            To create a CondaEnv, please use the factory methods :func:`env` or :func:`conda_env`.
        """
        self.reqs = reqs
        self.conda_yaml = conda_yaml  # dict representing conda env
        super().__init__(
            name=name,
            reqs=reqs,
            setup_cmds=setup_cmds,
            env_vars=env_vars,
            working_dir=working_dir,
            secrets=secrets,
            dryrun=dryrun,
        )

    def config(self, condensed=True):
        config = super().config(condensed)
        config.update({"conda_yaml": self.conda_yaml})
        return config

    @property
    def env_name(self):
        return self.conda_yaml["name"]

    def _create_conda_env(
        self, force: bool = False, cluster: "Cluster" = None, node: Optional[str] = None
    ):
        yaml_path = Path(ENVS_DIR) / f"{self.env_name}.yml"

        env_exists = (
            f"\n{self.env_name} "
            in run_setup_command("conda info --envs", cluster=cluster, node=node)[1]
        )
        run_setup_command(f"mkdir -p {ENVS_DIR}", cluster=cluster, node=node)
        yaml_exists = (
            (Path(ENVS_DIR).expanduser() / f"{self.env_name}.yml").exists()
            if not cluster
            else run_setup_command(f"ls {yaml_path}", cluster=cluster, node=node)[0]
            == 0
        )

        if force or not (yaml_exists and env_exists):
            # dump config into yaml file on cluster
            if not cluster:
                python_commands = "; ".join(
                    [
                        "import yaml",
                        "from pathlib import Path",
                        f"path = Path('{ENVS_DIR}').expanduser()",
                        f"yaml.dump({self.conda_yaml}, open(path / '{self.env_name}.yml', 'w'))",
                    ]
                )
                subprocess.run(f'python -c "{python_commands}"', shell=True)
            else:
                contents = yaml.dump(self.conda_yaml)
                run_setup_command(
                    f"echo $'{contents}' > {yaml_path}", cluster=cluster, node=node
                )

            # create conda env from yaml file
            run_setup_command(
                f"conda env create -f {yaml_path}", cluster=cluster, node=node
            )

            env_exists = (
                f"\n{self.env_name} "
                in run_setup_command("conda info --envs", cluster=cluster, node=node)[1]
            )
            if not env_exists:
                raise RuntimeError(f"conda env {self.env_name} not created properly.")

    def install(
        self, force: bool = False, cluster: "Cluster" = None, node: Optional[str] = None
    ):
        """Locally install packages and run setup commands.

        Args:
            force (bool, optional): Whether to force re-install env if it has already been installed.
                (default: ``False``)
            cluster (bool, optional): If None, installs env locally. Otherwise installs remotely
                on the cluster using SSH. (default: ``None``)
        """
        if not any(["python" in dep for dep in self.conda_yaml["dependencies"]]):
            status_codes = run_setup_command(
                "python --version", cluster=cluster, node=node
            )
            base_python_version = (
                status_codes[1].split()[1]
                if status_codes[0] == 0
                else CONDA_PREFERRED_PYTHON_VERSION
            )
            self.conda_yaml["dependencies"].append(f"python=={base_python_version}")
        install_conda(cluster=cluster)
        local_env_exists = (
            f"\n{self.env_name} "
            in run_setup_command("conda info --envs", cluster=cluster, node=node)[1]
        )

        # If we're doing the install remotely via SSH (e.g. for default_env), there is no cache
        if not cluster:
            # Hash the config_for_rns to check if we need to create/install the conda env
            env_config = self.config()
            # Remove the name because auto-generated names will be different, but the installed components are the same
            env_config.pop("name")
            install_hash = hash(str(env_config))
            # Check the existing hash
            if (
                local_env_exists
                and install_hash in obj_store.installed_envs
                and not force
            ):
                logger.debug("Env already installed, skipping")
                return
            obj_store.installed_envs[install_hash] = self.name

        self._create_conda_env(force=force, cluster=cluster, node=node)

        self._install_reqs(cluster=cluster, node=node)
        self._run_setup_cmds(cluster=cluster, node=node)

        return

    @property
    def _run_cmd(self):
        """Command prefix to run on Conda Env."""
        return f"conda run -n {self.env_name}"

    @property
    def _activate_cmd(self):
        """Command to activate Conda Env."""
        return f"conda activate {self.env_name}"
