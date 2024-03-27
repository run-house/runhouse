import copy
import json
import logging
import sys
from enum import Enum
from io import StringIO
from pathlib import Path
from typing import Any, List, Optional, Union

from runhouse.constants import LOGS_DIR
from runhouse.globals import configs, rns_client
from runhouse.resources.blobs import file

# Need to alias so it doesn't conflict with the folder property
from runhouse.resources.folders import Folder, folder as folder_factory
from runhouse.resources.hardware import _current_cluster, _get_cluster_from, Cluster
from runhouse.resources.resource import Resource
from runhouse.rns.top_level_rns_fns import resolve_rns_path
from runhouse.rns.utils.api import log_timestamp, resolve_absolute_path

# Load the root logger
logger = logging.getLogger("")


class RunStatus(str, Enum):
    NOT_STARTED = "NOT_STARTED"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    CANCELLED = "CANCELLED"
    ERROR = "ERROR"


class RunType(str, Enum):
    CMD_RUN = "CMD"
    FUNCTION_RUN = "FUNCTION"
    CTX_MANAGER = "CTX_MANAGER"


class Run(Resource):
    RESOURCE_TYPE = "run"

    LOCAL_RUN_PATH = f"{rns_client.rh_directory}/runs"

    RUN_CONFIG_FILE = "config_for_run.json"
    RESULT_FILE = "result.pkl"
    INPUTS_FILE = "inputs.pkl"

    def __init__(
        self,
        name: str = None,
        fn_name: str = None,
        cmds: list = None,
        log_dest: str = "file",
        path: str = None,
        system: Union[str, Cluster] = None,
        data_config: dict = None,
        status: RunStatus = RunStatus.NOT_STARTED,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        creator: Optional[str] = None,
        creation_stacktrace: Optional[str] = None,
        upstream_artifacts: Optional[List] = None,
        downstream_artifacts: Optional[List] = None,
        run_type: RunType = RunType.CMD_RUN,
        error: Optional[str] = None,
        error_traceback: Optional[str] = None,
        overwrite: bool = False,
        dryrun: bool = False,
        **kwargs,
    ):
        """
        Runhouse Run object

        .. note::
            To load an existing Run, please use the factory method :func:`run`.
        """
        run_name = name or str(self._current_timestamp())
        super().__init__(name=run_name, dryrun=dryrun)

        self.log_dest = log_dest
        self.folder = None
        if self.log_dest == "file":
            folder_system = system or Folder.DEFAULT_FS
            folder_path = (
                resolve_absolute_path(path)
                if path
                else (
                    self._base_local_folder_path(self.name)
                    if folder_system == Folder.DEFAULT_FS
                    else self._base_cluster_folder_path(name=run_name)
                )
            )

            if overwrite:
                # Delete the Run from the system if one already exists
                self._delete_existing_run(folder_path, folder_system)

            # Create new folder which lives on the system and contains all the Run's data:
            # (run config, stdout, stderr, inputs, result)
            self.folder = folder_factory(
                path=folder_path,
                system=folder_system,
                data_config=data_config,
                dryrun=dryrun,
            )

        self.status = status
        self.start_time = start_time
        self.end_time = end_time
        self.creator = creator
        self.creation_stacktrace = creation_stacktrace
        self.upstream_artifacts = upstream_artifacts or []
        self.downstream_artifacts = downstream_artifacts or []
        self.fn_name = fn_name
        self.cmds = cmds
        self.run_type = run_type or self._detect_run_type()
        self.error = error
        self.traceback = error_traceback
        # TODO string representation of inputs

    def __enter__(self):
        self.status = RunStatus.RUNNING
        self.start_time = self._current_timestamp()

        # Begin tracking the Run in the rns_client - this adds the current Run to the stack of active Runs
        rns_client.start_run(self)

        if self.log_dest == "file":
            # Capture stdout and stderr to the Run's folder
            self.folder.mkdir()
            # TODO fix the fact that we keep appending and then stream back the full file
            sys.stdout = StreamTee(sys.stdout, [Path(self._stdout_path).open(mode="a")])
            sys.stderr = StreamTee(sys.stderr, [Path(self._stderr_path).open(mode="a")])

            # Add the stdout and stderr handlers to the root logger
            self._stdout_handler = logging.StreamHandler(sys.stdout)
            logger.addHandler(self._stdout_handler)

        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.end_time = self._current_timestamp()
        if exc_type:
            self.status = RunStatus.ERROR
            self.error = exc_value
            self.traceback = exc_traceback
        else:
            self.status = RunStatus.COMPLETED

        # Pop the current Run from the stack of active Runs
        rns_client.stop_run()

        # if self.run_type == RunType.CMD_RUN:
        #     # Save Run config to its folder on the system - this will already happen on the cluster
        #     # for function based Runs
        #     self._write_config()
        #
        #     # For cmd runs we are using the SSH command runner to get the stdout / stderr
        #     return

        # TODO [DG->JL] Do we still need this?
        # stderr = f"{type(exc_value).__name__}: {str(exc_value)}" if exc_value else ""
        # self.write(data=stderr.encode(), path=self._stderr_path)

        if self.log_dest == "file":
            logger.removeHandler(self._stdout_handler)

            # Flush stdout and stderr
            # sys.stdout.flush()
            # sys.stderr.flush()

            # Restore stdout and stderr
            if hasattr(sys.stdout, "instream"):
                sys.stdout = sys.stdout.instream
            if hasattr(sys.stderr, "instream"):
                sys.stderr = sys.stderr.instream

            # Save Run config to its folder on the system - this will already happen on the cluster
            # for function based Runs
            # self._write_config()

        # return False to propagate any exception that occurred inside the with block
        return False

    @staticmethod
    def from_config(config: dict, dryrun=False):
        return Run(**config, dryrun=dryrun)

    def __getstate__(self):
        """Remove the folder object from the Run before pickling it."""
        state = self.__dict__.copy()
        state["folder"] = None
        state["_stdout_handler"] = None
        return state

    def config(self, condensed=True):
        """Metadata to store in RNS for the Run."""
        config = super().config(condensed)
        base_config = {
            "status": self.status,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "run_type": self.run_type,
            "log_dest": self.log_dest,
            "creator": self.creator,
            "fn_name": self.fn_name,
            "cmds": self.cmds,
            # NOTE: artifacts are currently only tracked in context manager based runs
            "upstream_artifacts": self.upstream_artifacts,
            "downstream_artifacts": self.downstream_artifacts,
            "path": self.folder.path,
            "system": self._resource_string_for_subconfig(
                self.folder.system, condensed
            ),
            "error": str(self.error),
            "traceback": str(self.traceback),
        }
        config.update(base_config)
        return config

    def populate_init_provenance(self):
        self.creator = configs.username
        self.creation_stacktrace = "".join(self.traceback.format_stack(limit=11)[1:])

    @property
    def run_config(self):
        """Config to save in the Run's dedicated folder on the system.
        Note: this is different from the config saved in RNS, which is the metadata for the Run.
        """
        config = {
            "name": self.name,
            "status": self.status,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "run_type": self.run_type,
            "fn_name": self.fn_name,
            "cmds": self.cmds,
            # NOTE: artifacts are currently only tracked in context manager based runs
            "upstream_artifacts": self.upstream_artifacts,
            "downstream_artifacts": self.downstream_artifacts,
        }
        return config

    def save(self, name: str = None, overwrite: bool = True, folder: str = None):
        """If the Run name is being overwritten (ex: initially created with auto-generated name),
        update the Run config stored on the system before saving to RNS."""
        config_for_rns = self.config()
        config_path = self._path_to_config()
        if not config_for_rns["name"] or name:
            config_for_rns["name"] = resolve_rns_path(name or self.name)
            self._write_config(config=config_for_rns)
            logger.debug(f"Updated Run config name in path: {config_path}")

        return super().save(name, overwrite, folder)

    def write(self, data: Any, path: str):
        """Write data (ex: function inputs or result, stdout, stderr) to the Run's dedicated folder on the system."""
        file(system=self.folder.system, path=path).write(data, serialize=False)

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
        run_config = self._load_run_config(folder=self.folder)
        # Need the metadata from RNS and the Run specific data in order to re-load the Run object
        config = {**self.config(), **run_config}
        return Run.from_config(config, dryrun=True)

    def inputs(self) -> bytes:
        """Load the pickled function inputs saved on the system for the Run."""
        return self._load_blob_from_path(path=self._fn_inputs_path()).fetch()

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
            return self._load_blob_from_path(path=results_path).fetch()
        elif run_status == RunStatus.ERROR:
            logger.debug("Run failed, returning stderr")
            return self.stderr()
        else:
            logger.debug(f"Run status: {self.status}, returning stdout")
            return self.stdout()

    def stdout(self) -> str:
        """Read the stdout saved on the system for the Run."""
        stdout_path = self._stdout_path
        logger.debug(f"Reading stdout from path: {stdout_path}")

        return self._load_blob_from_path(path=stdout_path).fetch().decode().strip()

    def stderr(self) -> str:
        """Read the stderr saved on the system for the Run."""
        stderr_path = self._stderr_path
        logger.debug(f"Reading stderr from path: {stderr_path}")

        return self._load_blob_from_path(stderr_path).fetch().decode().strip()

    def _fn_inputs_path(self) -> str:
        """Path to the pickled inputs used for the function which are saved on the system."""
        return f"{self.folder.path}/{self.INPUTS_FILE}"

    def _fn_result_path(self) -> str:
        """Path to the pickled result for the function which are saved on the system."""
        return f"{self.folder.path}/{self.RESULT_FILE}"

    def _load_blob_from_path(self, path: str):
        """Load a blob from the Run's folder in the specified path. (ex: function inputs, result, stdout, stderr)."""
        return file(path=path, system=self.folder.system)

    def _register_new_run(self):
        """Log a Run once it's been triggered on the system."""
        self.start_time = self._current_timestamp()
        self.status = RunStatus.RUNNING

        # Write config data for the Run to its config file on the system
        logger.debug(f"Registering new Run on system in path: {self.folder.path}")
        self._write_config()

    def _register_fn_run_completion(self, run_status: RunStatus):
        """Update a function based Run's config after its finished running on the system."""
        self.end_time = self._current_timestamp()
        self.status = run_status

        logger.debug(f"Registering a completed fn Run with status: {run_status}")
        self._write_config()

    def _register_cmd_run_completion(self, return_codes: list):
        """Update a cmd based Run's config and register its stderr and stdout after running on the system."""
        run_status = RunStatus.ERROR if return_codes[0][0] != 0 else RunStatus.COMPLETED
        self.status = run_status

        logger.debug(f"Registering a completed cmd Run with status: {run_status}")
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
        config_to_write = config or self.config()
        logger.debug(f"Config to save on system: {config_to_write}")
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

    @property
    def _stdout_path(self) -> str:
        """Path to the stdout file for the Run."""
        return self._path_to_file_by_ext(ext=".out")

    @property
    def _stderr_path(self) -> str:
        """Path to the stderr file for the Run."""
        return self._path_to_file_by_ext(ext=".err")

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

        existing_folder.rm(recursive=True)

    @staticmethod
    def _load_run_config(folder: Folder) -> dict:
        """Load the Run config file saved for the Run in its dedicated folder on the system ."""
        try:
            return json.loads(folder.get(Run.RUN_CONFIG_FILE))
        except FileNotFoundError:
            return {}

    @staticmethod
    def _base_cluster_folder_path(name: str):
        """Path to the base folder for this Run on a cluster."""
        return f"{LOGS_DIR}/{name}"

    @staticmethod
    def _base_local_folder_path(name: str):
        """Path to the base folder for this Run on a local system."""
        return f"{LOGS_DIR}/{name}"


class StreamTee(object):
    def __init__(self, instream, outstreams):
        self.instream = instream
        self.outstreams = outstreams

    def write(self, message):
        self.instream.write(message)
        for stream in self.outstreams:
            if message:
                stream.write(message)
                # We flush here to ensure that the logs are written to the file immediately
                # see https://github.com/run-house/runhouse/pull/724
                stream.flush()

    def writelines(self, lines):
        self.instream.writelines(lines)
        for stream in self.outstreams:
            stream.writelines(lines)
            stream.flush()

    def flush(self):
        self.instream.flush()
        for stream in self.outstreams:
            stream.flush()

    def __getattr__(self, item):
        # Needed in case someone calls a method on instream, such as Ray calling sys.stdout.istty()
        return getattr(self.instream, item)


class capture_stdout:
    """Context manager for capturing stdout to a file, list, or stream, while still printing to stdout."""

    def __init__(self, output=None):
        self.output = output
        self._stream = None

    def __enter__(self):
        if self.output is None:
            self.output = StringIO()

        if isinstance(self.output, str):
            self._stream = open(self.output, "w")
        else:
            self._stream = self.output
        sys.stdout = StreamTee(sys.stdout, [self])
        sys.stderr = StreamTee(sys.stderr, [self])
        return self

    def write(self, message):
        self._stream.write(message)

    def flush(self):
        self._stream.flush()

    @property
    def stream(self):
        if isinstance(self.output, str):
            return open(self.output, "r")
        return self._stream

    def list(self):
        if isinstance(self.output, str):
            return self.stream.readlines()
        return (self.stream.getvalue() or "").splitlines()

    def __str__(self):
        return self.stream.getvalue()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if hasattr(sys.stdout, "instream"):
            sys.stdout = sys.stdout.instream
        if hasattr(sys.stderr, "instream"):
            sys.stderr = sys.stderr.instream
        self._stream.close()
        return False


def run(
    name: str = None,
    log_dest: str = "file",
    path: str = None,
    system: Union[str, Cluster] = None,
    data_config: dict = None,
    load: bool = True,
    dryrun: bool = False,
    **kwargs,
) -> Union["Run", None]:
    """Constructs a Run object.

    Args:
        name (Optional[str]): Name of the Run to load.
        log_dest (Optional[str]): Whether to save the Run's logs to a file or stream them back. (Default: ``file``)
        path (Optional[str]): Path to the Run's dedicated folder on the system where the Run lives.
        system (Optional[str or Cluster]): File system or cluster name where the Run lives.
            If providing a file system this must be one of:
            [``file``, ``github``, ``sftp``, ``ssh``, ``s3``, ``gs``, ``azure``].
            We are working to add additional file system support.
        data_config (Optional[Dict]): The data config to pass to the underlying fsspec handler for the folder.
        load (bool): Whether to try reloading an existing Run from configs. (Default: ``True``)
        dryrun (bool): Whether to create the Blob if it doesn't exist, or load a Blob object as a dryrun.
            (Default: ``False``)
        **kwargs: Optional kwargs for the Run.

    Returns:
        Run: The loaded Run object.
    """
    if name and load and not any([path, system, data_config, kwargs]):
        # Try reloading existing Run from RNS
        return Run.from_name(name, dryrun=dryrun)

    if name and path is None and log_dest == "file":
        path = (
            Run._base_cluster_folder_path(name=name)
            if isinstance(system, Cluster)
            else Run._base_local_folder_path(name=name)
        )

    system = _get_cluster_from(
        system or _current_cluster(key="config") or Folder.DEFAULT_FS, dryrun=dryrun
    )

    run_obj = Run(
        name=name,
        log_dest=log_dest,
        path=path,
        system=system,
        data_config=data_config,
        dryrun=dryrun,
        **kwargs,
    )

    return run_obj
