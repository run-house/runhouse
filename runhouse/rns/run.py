import copy
import json
import logging
import sys
from datetime import datetime
from io import StringIO
from typing import Any, Callable, List, Optional, Union

from ray import cloudpickle as pickle

from runhouse import blob
from runhouse.rh_config import obj_store, rns_client
from runhouse.rns.api_utils.utils import log_timestamp
from runhouse.rns.folders.folder import Folder, folder
from runhouse.rns.function import Function
from runhouse.rns.hardware import Cluster
from runhouse.rns.resource import Resource
from runhouse.rns.top_level_rns_fns import resolve_rns_path

logger = logging.getLogger(__name__)


class Run(Resource):
    RESOURCE_TYPE = "run"

    LOCAL_RUN_PATH = f"{rns_client.rh_directory}/runs"
    RUN_CONFIG_FILE = "config_for_rns.json"

    RUNNING_STATUS = "RUNNING"
    COMPLETED_STATUS = "COMPLETED"
    ERROR_STATUS = "ERROR"

    # Supported Run types
    CLI_RUN = "CLI"
    FUNCTION_RUN = "FUNCTION"
    CTX_MANAGER = "CTX_MANAGER"

    def __init__(
        self,
        name: str = None,
        fn: Callable = None,
        cmds: list = None,
        dryrun: bool = False,
        status: str = None,
        start_time: int = None,
        end_time: int = None,
        overwrite: bool = False,
        **kwargs,
    ):
        """
        Runhouse Run object

        .. note::
                To build a Run, please use the factory method :func:`run`.
        """
        run_name = name or str(self.current_timestamp())
        super().__init__(name=run_name, dryrun=dryrun)

        self.cmds = cmds
        self.run_type = kwargs.get("run_type") or self._detect_run_type(fn)

        self.status = status
        self.start_time = start_time
        self.end_time = end_time

        folder_system = kwargs.get("system") or Folder.DEFAULT_FS

        folder_path = kwargs.get("path") or (
            self._base_local_folder_path()
            if folder_system == Folder.DEFAULT_FS
            else self.base_cluster_folder_path(name=run_name)
        )

        if overwrite:
            # Delete existing Run on the system before a new one is created
            existing_folder = folder(
                path=folder_path,
                system=folder_system,
                dryrun=True,
            )
            existing_folder.delete_in_system(path=folder_path)

        # Create new folder which lives on the system and contains all the Run's data:
        # (run config, stdout, stderr, inputs, result)
        self._folder = folder(
            path=folder_path,
            system=folder_system,
            dryrun=dryrun,
        )

        self.fn_name = kwargs.get("fn_name") or (fn.__name__ if fn else None)

        # Artifacts loaded by the Run (i.e. upstream dependencies)
        self.upstream_artifacts: list = kwargs.get("upstream_artifacts", [])

        # Artifacts saved by the Run (i.e. downstream dependencies)
        self.downstream_artifacts: list = kwargs.get("downstream_artifacts", [])

        self._stdout_path = self.path_to_stdout()
        self._stderr_path = self.path_to_stderr()

    def __enter__(self):
        self.status = self.RUNNING_STATUS
        self.start_time = self.current_timestamp()

        # Begin tracking the Run in the rns_client - this adds the current Run to the stack of active Runs
        rns_client.start_run(self)

        sys.stdout = StringIO()
        sys.stderr = StringIO()

        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.end_time = self.current_timestamp()
        self.status = self.COMPLETED_STATUS

        # Pop the current Run from the stack of active Runs
        rns_client.stop_run()

        # Save Run config to its folder on the system - this will already happen on the cluster
        # for function based runs
        self._write_config()

        if self.run_type == self.FUNCTION_RUN:
            # For function based Runs we use Ray to generate the .out and .err files on the cluster
            return

        stdout = sys.stdout.getvalue()
        stderr = sys.stderr.getvalue()

        # save stdout and stderr to their respective log files
        self.write(data=stdout.encode(), path=self._stdout_path)
        self.write(data=stderr.encode(), path=self._stderr_path)

        # return False to propagate any exception that occurred inside the with block
        return False

    @staticmethod
    def from_config(config: dict, dryrun=True):
        return Run(**config, dryrun=dryrun)

    @property
    def config_for_rns(self):
        config = super().config_for_rns
        base_config = {
            "path": self.path,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "resource_type": self.RESOURCE_TYPE,
            "system": self._resource_string_for_subconfig(self._folder.system),
            "status": self.status,
            # NOTE: artifacts are currently only tracked in context manager based runs
            "upstream_artifacts": self.upstream_artifacts,
            "downstream_artifacts": self.downstream_artifacts,
        }
        config.update({**base_config, **self.run_config})
        return config

    @property
    def path(self):
        return self._folder.path

    @property
    def run_config(self):
        if self.run_type == self.FUNCTION_RUN:
            # Function based run
            return {
                "fn_name": self.fn_name,
                "run_type": self.FUNCTION_RUN,
            }

        elif self.run_type == self.CLI_RUN:
            # CLI command based run
            return {
                "cmds": self.cmds,
                "run_type": self.CLI_RUN,
            }

        elif self.run_type == self.CTX_MANAGER:
            # Context manager based run
            return {
                "run_type": self.CTX_MANAGER,
            }

        else:
            raise TypeError(f"Unknown run type {self.run_type}")

    def register_upstream_artifact(self, artifact_name: str):
        """Track a Runhouse object loaded in the Run's context manager. This object's name
        will be saved to the upstream artifact registry of the Run's config."""
        if artifact_name not in self.upstream_artifacts:
            self.upstream_artifacts.append(artifact_name)

    def register_downstream_artifact(self, artifact_name: str):
        """Track a Runhouse object saved in the Run's context manager. This object's name
        will be saved to the downstream artifact registry of the Run's config."""
        if artifact_name not in self.downstream_artifacts:
            self.downstream_artifacts.append(artifact_name)

    def save(
        self,
        name: str = None,
        overwrite: bool = True,
    ):
        """If the Run name is being overwritten (ex: initially created with auto-generated name),
        update the Run config stored on the system before saving to RNS."""
        run_config = self.config_for_rns
        config_path = self.path_to_config()
        if not run_config["name"]:
            run_config["name"] = resolve_rns_path(name or self.name)
            self._write_config(config=run_config)
            logger.info(f"Updated Run config name in path: {config_path}")

        return super().save(name, overwrite)

    def write(self, data: Any, path: str):
        """Write Run associated data (ex: function inputs or result, stdout, stderr) to the
        specified path on the system."""
        blob(data=data, system=self._folder.system, path=path).write()

    def to(
        self, system, path: Optional[str] = None, data_config: Optional[dict] = None
    ):
        """Return a copy of the Run on the destination system and path."""
        new_run = copy.copy(self)

        for fp in [self._stdout_path, self._stderr_path]:
            # Stdout and Stderr files created on a cluster can be symlinks to the files that Ray creates
            # by default - before copying them to a new system  convert them to regular files
            self._convert_symlink_to_file(path=fp)

        if system == "here":
            # Save to default local path if none provided
            path = path or self._base_local_folder_path()

        new_run._folder = self._folder.to(
            system=system, path=path, data_config=data_config
        )

        return new_run

    def inputs(self) -> bytes:
        """Read the pickled function inputs saved on the system for the Run.
        Note: It's the user's responsibility to unpickle the inputs."""
        inputs_path = self.fn_inputs_path()
        if not self.exists_in_system(inputs_path):
            raise FileNotFoundError(
                f"No inputs found on {self._folder.system} in path: {inputs_path}"
            )

        return blob(path=inputs_path, system=self._folder.system).fetch()

    def result(self) -> Union[Any, None]:
        """Read and deserialize the function result saved on the system for the Run.
        If no result file is found, return None."""
        result_path = self.fn_result_path()
        if not self.exists_in_system(result_path):
            logger.warning(
                f"No result saved on {self._folder.system} in path: {result_path}. Please re-run, or "
                f"wait until the run has completed. (current status: {self.status})"
            )
            return None

        return pickle.loads(blob(path=result_path, system=self._folder.system).fetch())

    def stdout(self):
        """Read the stdout saved on the system for the Run."""
        stdout_path = self._stdout_path
        logger.info(f"Reading stdout from path: {stdout_path}")
        if not self.exists_in_system(stdout_path):
            raise FileNotFoundError(f"No stdout file found in path: {stdout_path}")

        return (
            blob(path=stdout_path, system=self._folder.system).fetch().decode().strip()
        )

    def stderr(self):
        """Read the stderr saved on the system for the Run."""
        stderr_path = self._stderr_path
        logger.info(f"Reading stderr from path: {stderr_path}")
        if not self.exists_in_system(stderr_path):
            raise FileNotFoundError(f"No stderr file found in path: {stderr_path}")

        return (
            blob(path=stderr_path, system=self._folder.system).fetch().decode().strip()
        )

    def path_to_stdout(self):
        """File path to stdout file for the Run."""
        return self.path_to_file_by_ext(ext=".out")

    def path_to_stderr(self):
        """File path to stderr file for the Run."""
        return self.path_to_file_by_ext(ext=".err")

    def path_to_config(self):
        """Path the main folder storing the metadata, inputs, and results for the Run saved on the system."""
        return f"{self.path}/{self.RUN_CONFIG_FILE}"

    def delete_in_system(self, path: str = None):
        """Delete a Run from its system. Deletes the specified folder and all its contents from the Run's file system"""
        path = path or self.path
        self._folder.delete_in_system(path)

    def exists_in_system(self, path: str = None) -> bool:
        """Checks whether a path exists in the Run's file system.
        Optionally provide a custom path to check, otherwise will default to the Run's folder path."""
        return self._folder.fsspec_fs.exists(path or self.path)

    def register_new_fn_run(self):
        """Log a function based Run once it's been triggered on the system."""
        self.start_time = self.current_timestamp()
        self.status = self.RUNNING_STATUS

        # Write config data for the Run to its config file on the system
        logger.info("Registering new function run")
        self._write_config()

    def register_fn_run_completion(self):
        """Update the Run's config once the function has finished running on the system."""
        self.end_time = self.current_timestamp()

        # TODO [JL] identify a failed run
        self.status = self.COMPLETED_STATUS

        logger.info("Registering function run completion")
        self._write_config()

    def _write_config(self, config: dict = None, overwrite: bool = True):
        """Write the Run's config data to the system (this is the same data that will be stored in RNS
        if the Run is saved)."""
        logger.info(f"Config to save on system: {self.config_for_rns}")
        self._folder.put(
            {self.RUN_CONFIG_FILE: config or self.config_for_rns},
            overwrite=overwrite,
            as_json=True,
        )

    def _detect_run_type(self, fn):
        if isinstance(fn, Callable):
            return self.FUNCTION_RUN
        elif self.cmds is not None:
            return self.CLI_RUN
        else:
            return self.CTX_MANAGER

    def path_to_file_by_ext(self, ext: str) -> str:
        """Path the file for the Run saved on the system for a provided extension (ex: ``.out`` or ``.err``)."""
        existing_file = self._find_file_path_by_ext(ext=ext)
        if existing_file:
            # If file already exists in file (ex: with function on a Ray cluster this will already be
            # generated for us)
            return existing_file

        path_to_ext = f"{self.path}/{self.name}" + ext
        return path_to_ext

    def _convert_symlink_to_file(self, path: str):
        """If the system is a Cluster and the file path is a symlink, convert it to a regular file.
        This is necessary to allow for copying of the file between systems (ex: cluster --> s3 or cluster --> local)."""
        if isinstance(self._folder.system, Cluster):
            status_codes: list = self._folder.system.run(
                [f"test -h {path} && echo True || echo False"], stream_logs=True
            )
            if status_codes[0][1].strip() == "True":
                # If it's a symlink convert it to a regular file
                self._folder.system.run(
                    [f"cp --remove-destination `readlink {path}` {path}"]
                )

    def _find_file_path_by_ext(self, ext: str) -> Union[str, None]:
        """Get the file path by provided extension. Needed when loading the stdout and stderr files associated
        with a particular run."""
        if not self.exists_in_system():
            # Folder not found on system
            return None

        folder_contents: list = self._folder.ls()
        if not folder_contents:
            return None

        files_with_ext = list(filter(lambda x: x.endswith(ext), folder_contents))
        if not files_with_ext:
            # No .out / .err file already created in the logs folder for this Run
            return None

        return files_with_ext[0]

    def _base_local_folder_path(self):
        """Path to the base folder for this Run on a local system."""
        return f"{self.LOCAL_RUN_PATH}/{self.name}"

    def fn_inputs_path(self):
        """Path the pickled inputs used for the function which are saved on the system."""
        return f"{self.path}/inputs.pkl"

    def fn_result_path(self):
        """Path the pickled result for the function which are saved on the system."""
        return f"{self.path}/result.pkl"

    @staticmethod
    def base_folder_name(name: str = None):
        """Name of the Run's parent folder which contains the Run's data (config, stdout, stderr, etc).
        If a name is provided, prepend that to the current timestamp to complete the folder name."""
        timestamp_key = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        if name is None:
            return timestamp_key
        return f"{name}_{timestamp_key}"

    @staticmethod
    def base_cluster_folder_path(name: str):
        """Path to the base folder for this Run on a cluster."""
        return f"~/{obj_store.LOGS_DIR}/{name}"

    @staticmethod
    def current_timestamp():
        return str(log_timestamp())

    @classmethod
    def create_run_name(cls, name_run: Union[str, bool], fn_name: str = None) -> str:
        """Generate the name for the Run. If a string is provided, use that as its name.
        Otherwise create one using the name (if provided) and the current timestamp."""
        if isinstance(name_run, str):
            return name_run
        elif name_run is True:
            return cls.base_folder_name(fn_name)
        else:
            raise TypeError("Invalid name_run type. Must be a string or `True`.")

    @classmethod
    def from_file(
        cls, run_name: str, folder_path: str = None, file_name: str = None
    ) -> "Run":
        """Load a local Run based on its name and path to its dedicated folder.

        Args:
            run_name (str): Name of the existing Run .
            folder_path (Optional[str]): Path to the Run's dedicated folder on the local file system.
                Defaults to the local path for Runs (in the rh folder of the working directory).
            file_name (Optional[str]): Name of the Run's config file. Defaults to the standard config file name.

        Returns:
            Run: The loaded Run object.
        """
        local_folder = folder(
            path=folder_path or f"{cls.LOCAL_RUN_PATH}/{run_name}",
            system=Folder.DEFAULT_FS,
            dryrun=True,
        )

        if not local_folder.exists_in_system():
            raise FileNotFoundError(f"No config found in path: {folder_path}")

        # Load config file for this Run
        run_config = json.loads(local_folder.get(name=file_name or cls.RUN_CONFIG_FILE))

        # Re-load the Run object
        return Run(**run_config, dryrun=True)

    @classmethod
    def path_to_latest_fn_run(cls, fn_name: str, system: str) -> Union[str, None]:
        """Path to the config file for the most recent based Run saved on the system for a given Function name."""
        system_folder = folder(
            path=obj_store.LOGS_DIR,
            system=system,
            dryrun=True,
        )

        latest_path = None
        most_recent_time = datetime.min

        # find the latest Run entry for this name
        for path in system_folder.ls():
            if fn_name not in path:
                continue

            try:
                timestamp_str = "_".join(path.split("_")[-2:])
                timestamp = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")

                if timestamp > most_recent_time:
                    latest_path = path
                    most_recent_time = timestamp

            except ValueError:
                # Path may not have a timestamp - if this is the only path use it, otherwise use the one with the
                # timestamp included as part of its run key
                if latest_path is None:
                    latest_path = path

        if latest_path is None:
            return None

        return f"~/{latest_path}/{cls.RUN_CONFIG_FILE}"

    @classmethod
    def default_config_path(cls, name: str, system: str):
        """Default path to the config file for a Run stored on a system."""
        if system == Folder.DEFAULT_FS:
            return f"{Run.LOCAL_RUN_PATH}/{name}/{cls.RUN_CONFIG_FILE}"

        # On a cluster save to the .rh logs folder
        return f"{Run.base_cluster_folder_path(name)}/{cls.RUN_CONFIG_FILE}"


def run(
    name: Optional[str] = None,
    fn: Optional[Union[str, Callable]] = None,
    cmds: Optional[List] = None,
    dryrun: bool = False,
    load: bool = True,
    overwrite: bool = False,
    **kwargs,
):
    """Returns a Run object, which can be used to capture logs, inputs and results, artifact info, etc.
    Currently supports running a particular Runhouse Function object or CLI command(s)
    on a specified system (ex: on a cluster).

    Args:
        name (Optional[str]): Name to give the Run object, to be reused later on.
        fn (Optional[str or Function]): The function to execute on the remote system when the function is called.
            Can be provided as a function name or as a Runhouse ``Function`` object.
        cmds (Optional[str]): List of CLI or Python commands to execute for the run.
        dryrun (bool): Whether to create the Run if it doesn't exist, or load a Run object as a dryrun.
            (Default: ``False``)
        load (bool): Whether to try loading an existing config for the Run from RNS. (Default: ``True``)
        overwrite (bool): Whether to overwrite the existing config for the Run. This will delete
            the run's dedicated folder on the system if it already exists.(Default: ``False``)

    Returns:
        Run: The resulting run.

    Example:
        >>> res = my_func(1, 2, name_run="my-run")

        >>> cpu = rh.cluster("^rh-cpu").up_if_not()
        >>> return_codes = cpu.run(["python --version"], name_run=True)

        >>> with rh.run(name="my-run"):
        >>>     print("do stuff")
    """
    config = rns_client.load_config(name) if load else {}

    name = name or config.get("name")
    fn = fn or config.get("fn_name")

    if isinstance(fn, str):
        fn = Function.from_name(fn)

    if fn is not None and not isinstance(fn, Callable):
        raise ValueError(f"fn provided must be a str or a Callable, not {type(fn)}")

    config["fn"] = fn
    config["cmds"] = cmds or config.get("cmds")
    config["name"] = name
    config["overwrite"] = config.get("overwrite") or overwrite

    new_run = Run.from_config({**config, **kwargs}, dryrun=dryrun)

    return new_run
