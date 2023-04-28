import logging
import subprocess
from pathlib import Path

from typing import Dict, List, Optional, Union

import yaml

from runhouse.rns.hardware import Cluster
from runhouse.rns.packages import Package

from .env import _process_reqs, Env


class CondaEnv(Env):
    RESOURCE_TYPE = "env"

    def __init__(
        self,
        conda_env: Union[str, Dict],
        name: Optional[str] = None,
        reqs: List[Union[str, Package]] = None,
        setup_cmds: List[str] = None,
        dryrun: bool = True,
        **kwargs,  # We have this here to ignore extra arguments when calling from_config
    ):
        """
        Runhouse CondaEnv object.

        .. note::
            To create a CondaEnv, please use the factory method :func:`env`.
        """
        reqs = _process_reqs(reqs)
        if isinstance(conda_env, str):
            if Path(conda_env).expanduser().exists():  # local yaml path
                conda_env = yaml.safe_load(open(conda_env))
            elif f"\n{conda_env} " in subprocess.check_output(
                "conda info --envs".split(" ")
            ).decode("utf-8"):
                # local env name, note that this only works if exporting/loading from same OS
                res = subprocess.check_output(
                    f"conda env export -n {conda_env}".split(" ")
                ).decode("utf-8")
                conda_env = yaml.load(res)
            else:
                raise Exception(
                    f"{conda_env} must be a Dict or point to an existing path or conda environment."
                )
        # ensure correct version to Ray -- impl is likely to change when SkyPilot adds support for other Ray versions
        conda_env["dependencies"] = (
            conda_env["dependencies"] if "dependencies" in conda_env else []
        )
        if not [dep for dep in conda_env["dependencies"] if "pip" in dep]:
            conda_env["dependencies"].append("pip")
        if not [
            dep
            for dep in conda_env["dependencies"]
            if isinstance(dep, Dict) and "pip" in dep
        ]:
            conda_env["dependencies"].append({"pip": ["ray==2.0.1"]})
        else:
            for dep in conda_env["dependencies"]:
                if (
                    isinstance(dep, Dict)
                    and "pip" in dep
                    and not [pip for pip in dep["pip"] if "ray" in pip]
                ):
                    dep["pip"].append("ray==2.0.1")
                    continue

        self.conda_env = conda_env  # dict representing conda env
        super().__init__(name=name, reqs=reqs, setup_cmds=setup_cmds, dryrun=dryrun)

    @staticmethod
    def from_config(config: dict, dryrun: bool = True):
        return CondaEnv(**config, dryrun=dryrun)

    @property
    def config_for_rns(self):
        config = super().config_for_rns
        config.update({"conda_env": self.conda_env})
        return config

    @property
    def env_name(self):
        return self.conda_env["name"]

    def _setup_env(self, system: Cluster):
        if not ["python" in dep for dep in self.conda_env["dependencies"]]:
            base_python_version = system.run(["python --version"])[0][1].split()[1]
            self.conda_env["dependencies"].append(f"python=={base_python_version}")

        try:
            system.run(["conda --version"])
        except FileNotFoundError:
            logging.info("Conda is not installed")
            system.run(
                "wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh "
                "-O ~/miniconda.sh".split(" ")
            )
            system.run(["bash ~/miniconda.sh -b -p ~/miniconda"])
            system.run(["source $HOME/miniconda3/bin/activate"])
            status = system.run("conda --version")[0]
            if status != 0:
                raise RuntimeError("Could not install Conda.")

        path = "~/.rh/envs"
        system.run([f"mkdir -p {path}"])
        system.run_python(
            [
                "import yaml",
                "from pathlib import Path",
                f"path = Path('{path}').expanduser()",
                f"yaml.dump({self.conda_env}, open(path/'{self.env_name}.yml', 'w'))",
            ]
        )

        if f"\n{self.env_name} " not in system.run(["conda info --envs"])[0][1]:
            system.run([f"conda env create -f {path}/{self.env_name}.yml"])
        # TODO [CC]: throw an error if environment is not constructed correctly
        system.run(['eval "$(conda shell.bash hook)"'])
        system.sync_runhouse_to_cluster(env=self)

        if self.reqs:
            system.install_packages(self.reqs, self._run_cmd)

        if self.setup_cmds:
            cmd = f"{self._activate_cmd} && {self.setup_cmds.join(' && ')}"
            system.run([cmd])

    @property
    def _run_cmd(self):
        """Command prefix to run on Conda Env."""
        return f"conda run -n {self.env_name}"

    @property
    def _activate_cmd(self):
        """Command to activate Conda Env."""
        return f"conda activate {self.env_name}"
