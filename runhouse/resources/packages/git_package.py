import logging
from pathlib import Path
from typing import Dict, Union

from runhouse.resources.envs.utils import run_setup_command

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

    def config(self, condensed: bool = True):
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

    # TODO for cluster
    def _install(self, env: Union[str, "Env"] = None, cluster: "Cluster" = None):
        from runhouse.resources.folders import Folder, folder

        if cluster and isinstance(self.install_target, str):
            install_target = folder(path=self.install_target, system=cluster)
        else:
            install_target = self.install_target

        # Clone down the repo
        if (cluster and not install_target.exists_in_system()) or (
            not cluster and not Path(self.install_target).exists()
        ):
            logging.info(f"Cloning: git clone {self.git_url}")
            run_setup_command(f"git clone {self.git_url}", cluster=cluster)
        else:
            install_path = (
                install_target.path
                if isinstance(install_target, Folder)
                else install_target
            )
            run_setup_command(
                f"git -C {install_path} fetch {self.git_url}", cluster=cluster
            )

        if self.revision:
            logging.info(f"Checking out revision: git checkout {self.revision}")
            run_setup_command(f"git -C {install_target} checkout {self.revision}")

        # Use super to install the package
        super()._install(env, cluster=cluster)

    @staticmethod
    def from_config(config: Dict, dryrun: bool = False, _resolve_children: bool = True):
        return GitPackage(**config, dryrun=dryrun)


def git_package(
    name: str = None,
    git_url: str = None,
    revision: str = None,
    install_method: str = None,
    install_str: str = None,
    load_from_den: bool = True,
    dryrun: bool = False,
):
    """
    Builds an instance of :class:`GitPackage`.

    Args:
        name (str, optional): Name to assign the package resource.
        git_url (str, optional): The GitHub URL of the package to install.
        revision (str, optional): Version of the Git package to install.
        install_method (str, optional): Method for installing the package. If left blank, defaults to
            local installation.
        install_str (str, optional): Additional arguments to add to installation command.
        load_from_den (bool, optional): Whether to try loading the package from Den. (Default: ``True``)
        dryrun (bool, optional): Whether to load the Package object as a dryrun, or create the Package if
            it doesn't exist. (Default: ``False``)

    Returns:
        GitPackage: The resulting GitHub Package.

    Example:
        >>> rh.git_package(git_url='https://github.com/runhouse/runhouse.git',
        >>>               install_method='pip', revision='v0.0.1')

    """
    if name and not any([install_method, install_str, git_url, revision]):
        # If only the name is provided and dryrun is set to True
        return Package.from_name(name, load_from_den=load_from_den, dryrun=dryrun)

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
