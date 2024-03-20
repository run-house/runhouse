import logging
import re
from pathlib import Path
from typing import Callable, List, Optional, Union

from runhouse.resources.envs import _get_env_from, Env
from runhouse.resources.functions.function import Function
from runhouse.resources.hardware import Cluster
from runhouse.resources.packages import git_package

logger = logging.getLogger(__name__)


def function(
    fn: Optional[Union[str, Callable]] = None,
    name: Optional[str] = None,
    system: Optional[Union[str, Cluster]] = None,  # deprecated
    env: Optional[Union[List[str], Env, str]] = None,
    dryrun: bool = False,
    load_secrets: bool = False,
    serialize_notebook_fn: bool = False,
):
    """runhouse.function(fn: str | Callable | None = None, name: str | None = None, system: str | Cluster | None = None, env: str | List[str] | Env | None = None, dryrun: bool = False, load_secrets: bool = False, serialize_notebook_fn: bool = False)

    Builds an instance of :class:`Function`.

    Args:
        fn (Optional[str or Callable]): The function to execute on the remote system when the function is called.
        name (Optional[str]): Name of the Function to create or retrieve.
            This can be either from a local config or from the RNS.
        env (Optional[List[str] or Env or str]): List of requirements to install on the remote cluster, or path to the
            requirements.txt file, or Env object or string name of an Env object.
        dryrun (bool): Whether to create the Function if it doesn't exist, or load the Function object as a dryrun.
            (Default: ``False``)
        load_secrets (bool): Whether or not to send secrets; only applicable if `dryrun` is set to ``False``.
            (Default: ``False``)
        serialize_notebook_fn (bool): If function is of a notebook setting, whether or not to serialized the function.
            (Default: ``False``)

    Returns:
        Function: The resulting Function object.

    Example:
        >>> import runhouse as rh

        >>> cluster = rh.ondemand_cluster(name="my_cluster")
        >>> def sum(a, b):
        >>>    return a + b

        >>> summer = rh.function(fn=sum, name="my_func").to(cluster, env=['requirements.txt']).save()

        >>> # using the function
        >>> res = summer(5, 8)  # returns 13

        >>> # Load function from above
        >>> reloaded_function = rh.function(name="my_func")
    """  # noqa: E501
    if name and not any([fn, system, env]):
        # Try reloading existing function
        return Function.from_name(name, dryrun)

    if system:
        raise Exception(
            "`system` argument is no longer supported in function factory function. "
            "Use `.to(system=system)` after construction to send the function to the system."
        )

    if not isinstance(env, Env):
        env = _get_env_from(env) or Env(working_dir="./", name=Env.DEFAULT_NAME)

    fn_pointers = None
    if callable(fn):
        fn_pointers = Function._extract_pointers(fn, reqs=env.reqs)
        if fn_pointers[1] == "notebook":
            fn_pointers = Function._handle_nb_fn(
                fn,
                fn_pointers=fn_pointers,
                serialize_notebook_fn=serialize_notebook_fn,
                name=fn_pointers[2] or name,
            )
    elif isinstance(fn, str):
        # Url must match a regex of the form
        # 'https://github.com/username/repo_name/blob/branch_name/path/to/file.py:func_name'
        # Use a regex to extract username, repo_name, branch_name, path/to/file.py, and func_name
        pattern = (
            r"https://github\.com/(?P<username>[^/]+)/(?P<repo_name>[^/]+)/blob/"
            r"(?P<branch_name>[^/]+)/(?P<path>[^:]+):(?P<func_name>.+)"
        )
        match = re.match(pattern, fn)

        if match:
            username = match.group("username")
            repo_name = match.group("repo_name")
            branch_name = match.group("branch_name")
            path = match.group("path")
            func_name = match.group("func_name")
        else:
            raise ValueError(
                "fn must be a callable or string of the form "
                '"https://github.com/username/repo_name/blob/branch_name/path/to/file.py:func_name"'
            )
        module_name = Path(path).stem
        relative_path = str(repo_name / Path(path).parent)
        fn_pointers = (relative_path, module_name, func_name)
        # TODO [DG] check if the user already added this in their reqs
        repo_package = git_package(
            git_url=f"https://github.com/{username}/{repo_name}.git",
            revision=branch_name,
        )
        env.reqs = [repo_package] + env.reqs

    new_function = Function(fn_pointers=fn_pointers, name=name, dryrun=dryrun, env=env)

    if load_secrets and not dryrun:
        new_function.send_secrets()

    return new_function
