import logging
import shlex
import subprocess
from pathlib import Path

from typing import Dict, List, Optional, Union

from runhouse.globals import obj_store

from runhouse.resources.packages import Package
from runhouse.utils import install_conda

from .env import Env


logger = logging.getLogger(__name__)


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

    @staticmethod
    def from_config(config: dict, dryrun: bool = True):
        return CondaEnv(**config, dryrun=dryrun)

    def config(self, condensed=True):
        config = super().config(condensed)
        config.update({"conda_yaml": self.conda_yaml})
        return config

    @property
    def env_name(self):
        return self.conda_yaml["name"]

    def _create_conda_env(self, force=False):
        path = "~/.rh/envs"
        subprocess.run(f"mkdir -p {path}", shell=True)

        local_env_exists = f"\n{self.env_name} " in subprocess.check_output(
            shlex.split("conda info --envs"), shell=False
        ).decode("utf-8")
        yaml_exists = (Path(path).expanduser() / f"{self.env_name}.yml").exists()

        if force or not (yaml_exists and local_env_exists):
            python_commands = "; ".join(
                [
                    "import yaml",
                    "from pathlib import Path",
                    f"path = Path('{path}').expanduser()",
                    f"yaml.dump({self.conda_yaml}, open(path / '{self.env_name}.yml', 'w'))",
                ]
            )
            subprocess.run(f'python -c "{python_commands}"', shell=True)

            if not local_env_exists:
                subprocess.run(
                    f"conda env create -f {path}/{self.env_name}.yml", shell=True
                )
                if f"\n{self.env_name} " not in subprocess.check_output(
                    shlex.split("conda info --envs"), shell=False
                ).decode("utf-8"):
                    raise RuntimeError(
                        f"conda env {self.env_name} not created properly."
                    )

    def install(self, force=False):
        """Locally install packages and run setup commands."""
        if not ["python" in dep for dep in self.conda_yaml["dependencies"]]:
            base_python_version = (
                subprocess.check_output(shlex.split("python --version"), shell=False)
                .decode("utf-8")
                .split()[1]
            )
            self.conda_yaml["dependencies"].append(f"python=={base_python_version}")
        install_conda()
        local_env_exists = f"\n{self.env_name} " in subprocess.check_output(
            shlex.split("conda info --envs"), shell=False
        ).decode("utf-8")

        # Hash the config_for_rns to check if we need to create/install the conda env
        env_config = self.config()
        # Remove the name because auto-generated names will be different, but the installed components are the same
        env_config.pop("name")
        install_hash = hash(str(env_config))
        # Check the existing hash
        if local_env_exists and install_hash in obj_store.installed_envs and not force:
            logger.debug("Env already installed, skipping")
            return
        obj_store.installed_envs[install_hash] = self.name

        self._create_conda_env()

        if self.reqs:
            for package in self.reqs:
                if isinstance(package, str):
                    pkg = Package.from_string(package)
                elif hasattr(package, "_install"):
                    pkg = package
                else:
                    raise ValueError(f"package {package} not recognized")

                logger.debug(f"Installing package: {str(pkg)}")
                pkg._install(self)

        return (
            self._run_command([f"{self.setup_cmds.join(' && ')}"])
            if self.setup_cmds
            else None
        )

    @property
    def _run_cmd(self):
        """Command prefix to run on Conda Env."""
        return f"conda run -n {self.env_name}"

    @property
    def _activate_cmd(self):
        """Command to activate Conda Env."""
        return f"conda activate {self.env_name}"
