import logging
import subprocess
from pathlib import Path

from runhouse import rh_config
from .package import Package


class GitPackage(Package):
    RESOURCE_TYPE = "package"

    def __init__(
        self,
        name: str = None,
        git_url: str = None,
        install_method: str = None,
        install_args: str = None,
        revision: str = None,
        dryrun: bool = False,
        **kwargs,  # We have this here to ignore extra arguments when calling from from_config
    ):
        """
        Runhouse Github Package resource.

        Example:
            >>> rh.GitPackage(git_url='https://github.com/runhouse/runhouse.git',
            >>>               install_method='pip', revision='v0.0.1')
        """
        super().__init__(
            name=name,
            dryrun=dryrun,
            install_method=install_method,
            install_target="./" + git_url.split("/")[-1].replace(".git", ""),
            install_args=install_args,
        )
        self.git_url = git_url
        self.revision = revision

    @property
    def config_for_rns(self):
        # If the package is just a simple Package.from_string string, no
        # need to store it in rns, just give back the string.
        # if self.install_method in ['pip', 'conda', 'git']:
        #     return f'{self.install_method}:{self.name}'
        config = super().config_for_rns
        self.save_attrs_to_config(config, ["git_url", "revision"])
        return config

    def __str__(self):
        if self.name:
            return f"GitPackage: {self.name}"
        return f"GitPackage: {self.git_url}@{self.revision}"

    def install(self):
        # Clone down the repo
        if not Path(self.install_target).exists():
            logging.info(f"Cloning: git clone {self.git_url}")
            subprocess.check_call(["git", "clone", self.git_url])
        else:
            logging.info(f"Pulling: git -C {self.install_target} fetch {self.git_url}")
            subprocess.check_call(
                f"git -C {self.install_target} fetch {self.git_url}".split(" ")
            )
        # Checkout the revision
        if self.revision:
            logging.info(f"Checking out revision: git checkout {self.revision}")
            subprocess.check_call(
                ["git", "-C", self.install_target, "checkout", self.revision]
            )
        # Use super to install the package
        super().install()

    @staticmethod
    def from_config(config: dict, dryrun=False):
        return GitPackage(**config, dryrun=dryrun)


def git_package(
    name=None,
    git_url: str = None,
    revision: str = None,
    install_method: str = None,
    install_str: str = None,
    dryrun=False,
):
    config = rh_config.rns_client.load_config(name)
    config["name"] = name or config.get("rns_address", None) or config.get("name")

    config["install_method"] = install_method or config.get("install_method", "local")
    config["install_args"] = install_str or config.get("install_args")
    config["revision"] = revision or config.get("revision")
    if git_url is not None:
        if not git_url.endswith(".git"):
            git_url += ".git"
        config["git_url"] = git_url

    new_package = GitPackage.from_config(config, dryrun=dryrun)

    return new_package
