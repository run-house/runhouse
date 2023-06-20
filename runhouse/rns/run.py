import copy
import json
import logging
import sys
from datetime import datetime
from enum import Enum
from io import StringIO
from typing import Any, Optional, Union

from ray import cloudpickle as pickle

from runhouse import blob
from runhouse.rh_config import obj_store, rns_client
from runhouse.rns.api_utils.utils import log_timestamp

# Need to alias so it doesn't conflict with the folder property
from runhouse.rns.folders import Folder, folder as folder_factory
from runhouse.rns.hardware import Cluster
from runhouse.rns.resource import Resource
from runhouse.rns.top_level_rns_fns import resolve_rns_path

logger = logging.getLogger(__name__)


class RunStatus(str, Enum):
    NOT_STARTED = "NOT_STARTED"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    ERROR = "ERROR"


class RunType(str, Enum):
    CMD_RUN = "CMD"
    FUNCTION_RUN = "FUNCTION"
    CTX_MANAGER = "CTX_MANAGER"


class Run(Resource):
    RESOURCE_TYPE = "run"

    LOCAL_RUN_PATH = f"{rns_client.rh_directory}/runs"

    RUN_CONFIG_FILE = "config_for_rns.json"
    RESULT_FILE = "result.pkl"
    INPUTS_FILE = "inputs.pkl"

    def __init__(
        self,
        name: str = None,
        fn_name: str = None,
        cmds: list = None,
        dryrun: bool = False,
        overwrite: bool = False,
        data_config: dict = None,
        **kwargs,
    ):
        """
        Runhouse Run object

        .. note::
            To load an existing Run, please use the factory method :func:`run`.
        """
        run_name = name or str(self._current_timestamp())
        super().__init__(name=run_name, dryrun=dryrun)

        self.cmds = cmds

        self.status = kwargs.get("status") or RunStatus.NOT_STARTED
        self.start_time = kwargs.get("start_time")
        self.end_time = kwargs.get("end_time")

        folder_system = kwargs.get("system") or Folder.DEFAULT_FS

        folder_path = kwargs.get("path") or (
            self._base_local_folder_path(self.name)
            if folder_system == Folder.DEFAULT_FS
            else self._base_cluster_folder_path(name=run_name)
        )

        if overwrite:
            self._delete_existing_run(folder_path, folder_system)

        # Create new folder which lives on the system and contains all the Run's data:
        # (run config, stdout, stderr, inputs, result)
        self.folder = folder_factory(
            path=folder_path,
            system=folder_system,
            data_config=data_config,
            dryrun=dryrun,
        )

        self.fn_name = fn_name or kwargs.get("fn_name")
        self.run_type = kwargs.get("run_type") or self._detect_run_type()

        # Artifacts loaded by the Run (i.e. upstream dependencies)
        self.upstream_artifacts: list = kwargs.get("upstream_artifacts", [])

        # Artifacts saved by the Run (i.e. downstream dependencies)
        self.downstream_artifacts: list = kwargs.get("downstream_artifacts", [])

        self._stdout_path = self._path_to_file_by_ext(ext=".out")
        self._stderr_path = self._path_to_file_by_ext(ext=".err")

    def __enter__(self):
        self.status = RunStatus.RUNNING
        self.start_time = self._current_timestamp()

        # Begin tracking the Run in the rns_client - this adds the current Run to the stack of active Runs
        rns_client.start_run(self)

        sys.stdout = StringIO()
        sys.stderr = StringIO()

        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.end_time = self._current_timestamp()
        self.status = RunStatus.COMPLETED

        # Pop the current Run from the stack of active Runs
        rns_client.stop_run()

        # Save Run config to its folder on the system - this will already happen on the cluster
        # for function based Runs
        self._write_config()

        if self.run_type in [RunType.FUNCTION_RUN, RunType.CMD_RUN]:
            # For function based Runs we use the logfiles already generated for the current Ray worker
            # on the cluster, and for cmd runs we are using the SSH command runner to get the stdout / stderr
            return

        stdout = sys.stdout.getvalue()
        stderr = sys.stderr.getvalue()

        # save stdout and stderr to their respective log files
        self.write(data=stdout.encode(), path=self._stdout_path)
        self.write(data=stderr.encode(), path=self._stderr_path)

        # return False to propagate any exception that occurred inside the with block
        return False

    @staticmethod
    def from_config(config: dict, dryrun=False):
        return Run(**config, dryrun=dryrun)

    @property
    def config_for_rns(self):
        config = super().config_for_rns
        base_config = {
            "path": self.folder.path,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "resource_type": self.RESOURCE_TYPE,
            "system": self._resource_string_for_subconfig(self.folder.system),
            "status": self.status,
            # NOTE: artifacts are currently only tracked in context manager based runs
            "upstream_artifacts": self.upstream_artifacts,
            "downstream_artifacts": self.downstream_artifacts,
        }
        config.update({**base_config, **self.run_config})
        return config

    @property
    def run_config(self):
        if self.run_type == RunType.FUNCTION_RUN:
            # Function based run
            return {
                "fn_name": self.fn_name,
                "run_type": RunType.FUNCTION_RUN,
            }

        elif self.run_type == RunType.CMD_RUN:
            # CLI command based run
            return {
                "cmds": self.cmds,
                "run_type": RunType.CMD_RUN,
            }

        elif self.run_type == RunType.CTX_MANAGER:
            # Context manager based run
            return {
                "run_type": RunType.CTX_MANAGER,
            }

        else:
            raise TypeError(f"Unknown run type {self.run_type}")

    def save(
        self,
        name: str = None,
        overwrite: bool = True,
    ):
        """If the Run name is being overwritten (ex: initially created with auto-generated name),
        update the Run config stored on the system before saving to RNS."""
        run_config = self.config_for_rns
        config_path = self._path_to_config()
        if not run_config["name"] or name:
            run_config["name"] = resolve_rns_path(name or self.name)
            self._write_config(config=run_config)
            logger.info(f"Updated Run config name in path: {config_path}")

        return super().save(name, overwrite)

    def write(self, data: Any, path: str):
        """Write data (ex: function inputs or result, stdout, stderr) to the Run's dedicated folder on the system."""
        blob(data=data, system=self.folder.system, path=path).write()

    def to(
        self,
        system,
        path: Optional[str] = None,
        data_config: Optional[dict] = None,
    ):
        """Send a Run to another system.

        Args:
            system (Union[str or Cluster]): Name of the system or Cluster object to copy the Run to.
            path (Optional[str]): Path to the on the system to save the Run.
                Defaults to the local path for Runs (in the rh folder of the working directory).
            data_config (Optional[dict]): Config to pass into fsspec handler for copying the Run.

        Returns:
            Run: A copy of the Run on the destination system and path.
        """
        # TODO: [JL] - support for `on_completion` (wait to copy the results to destination until async run completes)

        new_run = copy.copy(self)

        if self.run_type == RunType.FUNCTION_RUN:
            results_path = self._fn_result_path()
            # Pickled function result should be saved down to the Run's folder on the cluster
            if results_path not in self.folder.ls():
                raise FileNotFoundError(
                    f"No results saved down in path: {results_path}"
                )

        for fp in [self._stdout_path, self._stderr_path]:
            # Stdout and Stderr files created on a cluster can be symlinks to the files that we create via Ray
            # by default - before copying them to a new system make sure they are regular files
            self._convert_symlink_to_file(path=fp)

        if system == "here":
            # Save to default local path if none provided
            path = path or self._base_local_folder_path(self.name)

        new_run.folder = self.folder.to(
            system=system, path=path, data_config=data_config
        )

        return new_run

    def refresh(self) -> "Run":
        """Reload the Run object from the system. This is useful for checking the status of a Run.
        For example: ``my_run.refresh().status``"""
        run_config = json.loads(self.folder.get(name=self.RUN_CONFIG_FILE))
        return Run.from_config(run_config, dryrun=True)

    def inputs(self) -> bytes:
        """Load the pickled function inputs saved on the system for the Run."""
        return pickle.loads(
            self._load_blob_from_path(path=self._fn_inputs_path()).fetch()
        )

    def result(self):
        """Load the function result saved on the system for the Run. If the Run has failed return the stderr,
        otherwise return the stdout."""
        run_status = self.refresh().status
        if run_status == RunStatus.COMPLETED:
            results_path = self._fn_result_path()
            if results_path not in self.folder.ls():
                raise FileNotFoundError(
                    f"No results file found in path: {results_path}"
                )
            return pickle.loads(self._load_blob_from_path(path=results_path).fetch())
        elif run_status == RunStatus.ERROR:
            logger.info("Run failed, returning stderr")
            return self.stderr()
        else:
            logger.info(f"Run status: {self.status}, returning stdout")
            return self.stdout()

    def stdout(self) -> str:
        """Read the stdout saved on the system for the Run."""
        stdout_path = self._stdout_path
        logger.info(f"Reading stdout from path: {stdout_path}")

        return self._load_blob_from_path(path=stdout_path).fetch().decode().strip()

    def stderr(self) -> str:
        """Read the stderr saved on the system for the Run."""
        stderr_path = self._stderr_path
        logger.info(f"Reading stderr from path: {stderr_path}")

        return self._load_blob_from_path(stderr_path).fetch().decode().strip()

    def _fn_inputs_path(self) -> str:
        """Path to the pickled inputs used for the function which are saved on the system."""
        return f"{self.folder.path}/{self.INPUTS_FILE}"

    def _fn_result_path(self) -> str:
        """Path to the pickled result for the function which are saved on the system."""
        return f"{self.folder.path}/{self.RESULT_FILE}"

    def _load_blob_from_path(self, path: str):
        """Load a blob from the Run's folder in the specified path. (ex: function inputs, result, stdout, stderr)."""
        return blob(path=path, system=self.folder.system)

    def _register_new_run(self):
        """Log a Run once it's been triggered on the system."""
        self.start_time = self._current_timestamp()
        self.status = RunStatus.RUNNING

        # Write config data for the Run to its config file on the system
        logger.info("Registering new Run on system")
        self._write_config()

    def _register_fn_run_completion(self, run_status: RunStatus):
        """Update a function based Run's config after its finished running on the system."""
        self.end_time = self._current_timestamp()
        self.status = run_status

        logger.info(f"Registering a completed fn Run with status: {run_status}")
        self._write_config()

    def _register_cmd_run_completion(self, return_codes: list):
        """Update a cmd based Run's config and register its stderr and stdout after running on the system."""
        run_status = RunStatus.ERROR if return_codes[0][0] != 0 else RunStatus.COMPLETED
        self.status = run_status

        logger.info(f"Registering a completed cmd Run with status: {run_status}")
        self._write_config()

        # Write the stdout and stderr of the commands Run to the Run's folder
        self.write(data=return_codes[0][1].encode(), path=self._stdout_path)
        self.write(data=return_codes[0][2].encode(), path=self._stderr_path)

    def _write_config(self, config: dict = None, overwrite: bool = True):
        """Write the Run's config data to the system.

        Args:
            config (Optional[Dict]): Config to write. If none is provided, the Run's config for RNS will be used.
            overwrite (Optional[bool]): Overwrite the config if one is already saved down. Defaults to ``True``.
        """
        config_to_write = config or self.config_for_rns
        logger.info(f"Config to save on system: {config_to_write}")
        self.folder.put(
            {self.RUN_CONFIG_FILE: config_to_write},
            overwrite=overwrite,
            mode="w",
            write_fn=lambda data, f: json.dump(data, f, indent=4),
        )

    def _detect_run_type(self):
        if self.fn_name:
            return RunType.FUNCTION_RUN
        elif self.cmds is not None:
            return RunType.CMD_RUN
        else:
            return RunType.CTX_MANAGER

    def _path_to_config(self) -> str:
        """Path the main folder storing the metadata, inputs, and results for the Run saved on the system."""
        return f"{self.folder.path}/{self.RUN_CONFIG_FILE}"

    def _path_to_file_by_ext(self, ext: str) -> str:
        """Path the file for the Run saved on the system for a provided extension (ex: ``.out`` or ``.err``)."""
        existing_file = self._find_file_path_by_ext(ext=ext)
        if existing_file:
            # If file already exists in file (ex: with function on a Ray cluster this will already be
            # generated for us)
            return existing_file

        path_to_ext = f"{self.folder.path}/{self.name}" + ext
        return path_to_ext

    def _convert_symlink_to_file(self, path: str):
        """If the system is a Cluster and the file path is a symlink, convert it to a regular file.
        This is necessary to allow for copying of the file between systems (ex: cluster --> s3 or cluster --> local)."""
        if isinstance(self.folder.system, Cluster):
            status_codes: list = self.folder.system.run(
                [f"test -h {path} && echo True || echo False"], stream_logs=True
            )
            if status_codes[0][1].strip() == "True":
                # If it's a symlink convert it to a regular file
                self.folder.system.run(
                    [f"cp --remove-destination `readlink {path}` {path}"]
                )

    def _find_file_path_by_ext(self, ext: str) -> Union[str, None]:
        """Get the file path by provided extension. Needed when loading the stdout and stderr files associated
        with a particular run."""
        try:
            folder_contents: list = self.folder.ls(sort=True)
        except FileNotFoundError:
            return None

        files_with_ext = self._filter_files_by_ext(folder_contents, ext)
        if not files_with_ext:
            # No .out / .err file already created in the logs folder for this Run
            return None

        # Return the most recent file with this extension
        return files_with_ext[0]

    def _register_upstream_artifact(self, artifact_name: str):
        """Track a Runhouse object loaded in the Run's context manager. This object's name
        will be saved to the upstream artifact registry of the Run's config."""
        if artifact_name not in self.upstream_artifacts:
            self.upstream_artifacts.append(artifact_name)

    def _register_downstream_artifact(self, artifact_name: str):
        """Track a Runhouse object saved in the Run's context manager. This object's name
        will be saved to the downstream artifact registry of the Run's config."""
        if artifact_name not in self.downstream_artifacts:
            self.downstream_artifacts.append(artifact_name)

    @staticmethod
    def _current_timestamp():
        return str(log_timestamp())

    @staticmethod
    def _filter_files_by_ext(files: list, ext: str):
        return list(filter(lambda x: x.endswith(ext), files))

    @staticmethod
    def _delete_existing_run(folder_path, folder_system: str):
        """Delete existing Run on the system before a new one is created."""
        existing_folder = folder_factory(
            path=folder_path,
            system=folder_system,
        )

        existing_folder.rm()

    @staticmethod
    def _create_new_run_name(name: str = None) -> str:
        """Name of the Run's parent folder which contains the Run's data (config, stdout, stderr, etc).
        If a name is provided, prepend that to the current timestamp to complete the folder name."""
        timestamp_key = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        if name is None:
            return timestamp_key
        return f"{name}_{timestamp_key}"

    @staticmethod
    def _base_cluster_folder_path(name: str):
        """Path to the base folder for this Run on a cluster."""
        return f"{obj_store.LOGS_DIR}/{name}"

    @staticmethod
    def _base_local_folder_path(name: str):
        """Path to the base folder for this Run on a local system."""
        return f"{Run.LOCAL_RUN_PATH}/{name}"


def run(
    name: str = None,
    path: str = None,
    system: Union[str, Cluster] = None,
    data_config: dict = None,
) -> Union["Run", None]:
    """Load a Run based on the path to its dedicated folder on a system.

    Args:
        name (Optional[str]): Name of the Run to load.
        path (Optional[str]): Path to the Run's dedicated folder on the system where the Run lives.
        system (Optional[str or Cluster]): Name of the system or a cluster object where the Run lives.
        data_config (Optional[Dict]): The data config to pass to the underlying fsspec handler for the folder.

    Returns:
        Run: The loaded Run object.
    """
    if name is not None:
        path = (
            Run._base_cluster_folder_path(name=name)
            if isinstance(system, Cluster)
            else Run._base_local_folder_path(name=name)
        )

    if path is None:
        raise ValueError("Must provide either a name or path to load a Run.")

    system_folder = folder_factory(
        path=path,
        system=system,
        data_config=data_config,
    )

    if not system_folder.exists_in_system():
        logger.info(f"No Run config found in path: {system_folder.path}")
        return None

    # Load config file for this Run
    run_config = json.loads(system_folder.get(name=Run.RUN_CONFIG_FILE))

    # Re-load the Run object
    logger.info(f"Run config from path: {run_config}")
    return Run.from_config(run_config)
