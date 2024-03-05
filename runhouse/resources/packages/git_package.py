import logging
import subprocess
from pathlib import Path
from typing import Union

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

        .. note::
            To create a git package, please use the factory method :func:`git_package` or :func:`package`.
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

    def config(self, condensed=True):
        # If the package is just a simple Package.from_string string, no
        # need to store it in rns, just give back the string.
        # if self.install_method in ['pip', 'conda', 'git']:
        #     return f'{self.install_method}:{self.name}'
        config = super().config(condensed)
        self.save_attrs_to_config(config, ["git_url", "revision"])
        return config

    def __str__(self):
        if self.name:
            return f"GitPackage: {self.name}"
        return f"GitPackage: {self.git_url}@{self.revision}"

    def _install(self, env: Union[str, "Env"] = None):
        # Clone down the repo
        if not Path(self.install_target).exists():
            logging.info(f"Cloning: git clone {self.git_url}")
            subprocess.run(
                ["git", "clone", self.git_url],
                cwd=Path(self.install_target).expanduser().parent,
                check=True,
            )
        else:
            logging.info(f"Pulling: git -C {self.install_target} fetch {self.git_url}")
            subprocess.run(
                f"git -C {self.install_target} fetch {self.git_url}".split(" "),
                check=True,
                cwd=Path(self.install_target).expanduser().parent,
            )
        # Checkout the revision
        if self.revision:
            logging.info(f"Checking out revision: git checkout {self.revision}")
            subprocess.run(
                ["git", "-C", self.install_target, "checkout", self.revision],
                cwd=Path(self.install_target).expanduser().parent,
                check=True,
            )
        # Use super to install the package
        super()._install(env)

    @staticmethod
    def from_config(config: dict, dryrun=False):
        return GitPackage(**config, dryrun=dryrun)


def git_package(
    name: str = None,
    git_url: str = None,
    revision: str = None,
    install_method: str = None,
    install_str: str = None,
    dryrun: bool = False,
):
    """
    Builds an instance of :class:`GitPackage`.

    Args:
        name (str): Name to assign the package resource.
        git_url (str): The GitHub URL of the package to install.
        revision (str): Version of the Git package to install.
        install_method (str): Method for installing the package. If left blank, defaults to local installation.
        install_str (str): Additional arguments to add to installation command.
        dryrun (bool): Whether to load the Package object as a dryrun, or create the Package if it doesn't exist.
            (Default: ``False``)

    Returns:
        GitPackage: The resulting GitHub Package.

    Example:
        >>> rh.git_package(git_url='https://github.com/runhouse/runhouse.git',
        >>>               install_method='pip', revision='v0.0.1')

    """
    if name and not any([install_method, install_str, git_url, revision]):
        # If only the name is provided and dryrun is set to True
        return Package.from_name(name, dryrun)

    install_method = install_method or "local"
    if git_url is not None:
        if not git_url.endswith(".git"):
            git_url += ".git"

    return GitPackage(
        git_url=git_url,
        revision=revision,
        install_method=install_method,
        install_args=install_str,
        dryrun=dryrun,
    )
