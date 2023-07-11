import logging
from pathlib import Path

from typing import Dict, List, Optional, Union

from runhouse.rns.hardware import Cluster
from runhouse.rns.packages import Package

from .env import Env


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
            dryrun=dryrun,
        )

    @staticmethod
    def from_config(config: dict, dryrun: bool = True):
        return CondaEnv(**config, dryrun=dryrun)

    @property
    def config_for_rns(self):
        config = super().config_for_rns
        config.update({"conda_yaml": self.conda_yaml})
        return config

    @property
    def env_name(self):
        return self.conda_yaml["name"]

    def _setup_env(self, system: Cluster):
        if not ["python" in dep for dep in self.conda_yaml["dependencies"]]:
            base_python_version = system.run(["python --version"])[0][1].split()[1]
            self.conda_yaml["dependencies"].append(f"python=={base_python_version}")

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
                f"yaml.dump({self.conda_yaml}, open(path/'{self.env_name}.yml', 'w'))",
            ]
        )

        if f"\n{self.env_name} " not in system.run(["conda info --envs"])[0][1]:
            system.run([f"conda env create -f {path}/{self.env_name}.yml"])
        # TODO [CC]: throw an error if environment is not constructed correctly
        system.run(['eval "$(conda shell.bash hook)"'])
        system._sync_runhouse_to_cluster(env=self)

        if self.reqs:
            system.install_packages(self.reqs, self)

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
