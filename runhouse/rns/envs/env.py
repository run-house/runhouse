from typing import List, Optional, Union

from runhouse.rh_config import rns_client
from runhouse.rns.hardware import Cluster
from runhouse.rns.packages import Package
from runhouse.rns.resource import Resource


def process_reqs(reqs):
    preprocessed_reqs = []
    for req in reqs:
        # TODO [DG] the following is wrong. RNS address doesn't have to start with '/'. However if we check if each
        #  string exists in RNS this will be incredibly slow, so leave it for now.
        if isinstance(req, str) and req[0] == "/" and rns_client.exists(req):
            # If req is an rns address
            req = rns_client.load_config(req)
        preprocessed_reqs.append(req)
    return preprocessed_reqs


class Env(Resource):
    RESOURCE_TYPE = "env"

    def __init__(
        self,
        name: Optional[str] = None,
        reqs: List[Union[str, Package]] = None,
        setup_cmds: List[str] = None,
        dryrun: bool = True,
        **kwargs,  # We have this here to ignore extra arguments when calling from_config
    ):
        """
        Runhouse Env object.

        .. note::
            To create an Env, please use the factory method :func:`env`.
        """
        super().__init__(name=name, dryrun=dryrun)
        self.reqs = process_reqs(reqs)
        self.setup_cmds = setup_cmds

    @staticmethod
    def from_config(config: dict, dryrun: bool = True):
        """Create an Env object from a config dict"""
        return Env(**config, dryrun=dryrun)

    @property
    def config_for_rns(self):
        config = super().config_for_rns
        config.update(
            {
                "reqs": [
                    self._resource_string_for_subconfig(package)
                    for package in self.reqs
                ],
                "setup_cmds": self.setup_cmds,
            }
        )
        return config

    def to(self, system: Union[str, Cluster]):
        # env doesn't have a concept of system, so this just sets up the environment on the given cluster
        system.install_packages(self.reqs)

        if self.setup_cmds:
            system.run(self.setup_cmds)


def env(
    reqs: List[Union[str, Package]] = None,
    name: Optional[str] = None,
    setup_cmds: List[str] = None,
    dryrun: bool = True,
    load: bool = True,
):
    """Factory method for creating a Runhouse Env object.

    Args:
        reqs (List[str]): List of package names to install in this environment.
        conda_env (Union[str, Dict], optional): Path to a conda environment file or Dict representing conda env.
        name (Optional[str], optional): Name of the environment.
        dryrun (bool, optional): Whether to run in dryrun mode. (Default: ``True``)
        load (bool, optional): Whether to load the environment. (Default: ``True``)

    Returns:
        Env: The resulting Env object.

    Example:
        >>> # regular python env
        >>> env = rh.env(reqs=["pytorch", "pip"])
        >>> env = rh.env(reqs=["requirements.txt"], name="myenv")
    """

    config = rns_client.load_config(name) if load else {}
    config["name"] = name or config.get("rns_address", None) or config.get("name")
    config["reqs"] = reqs if reqs is not None else config.get("reqs", [])
    config["setup_cmds"] = (
        setup_cmds if setup_cmds is not None else config.get("setup_cmds")
    )

    return Env.from_config(config, dryrun=dryrun)
