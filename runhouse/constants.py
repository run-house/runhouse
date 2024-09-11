import os
import sys
from pathlib import Path
from typing import List

RESERVED_SYSTEM_NAMES: List[str] = ["file", "s3", "gs", "azure", "here"]
CLUSTER_CONFIG_PATH: str = "~/.rh/cluster_config.json"
CONFIG_YAML_PATH: str = "~/.rh/config.yaml"
SERVER_LOGFILE_PATH = "~/.rh/server.log"
LOCALHOST: str = "127.0.0.1"
LOCAL_HOSTS: List[str] = ["localhost", LOCALHOST]
TUNNEL_TIMEOUT = 5
NUM_PORTS_TO_TRY = 10

LOGS_DIR = ".rh/logs"
RH_LOGFILE_PATH = Path.home() / LOGS_DIR

ENVS_DIR = "~/.rh/envs"

GIGABYTE = 1024**3
MAX_MESSAGE_LENGTH = 1 * GIGABYTE

CLI_RESTART_CMD = "runhouse restart"
CLI_STOP_CMD = "runhouse stop"

DEFAULT_SERVER_PORT = 32300
DEFAULT_HTTPS_PORT = 443
DEFAULT_HTTP_PORT = 80
DEFAULT_SSH_PORT = 22

DEFAULT_RAY_PORT = 6379


DEFAULT_SERVER_HOST = "0.0.0.0"

LOGGING_WAIT_TIME = 0.5

# Commands
SERVER_START_CMD = f"{sys.executable} -m runhouse.servers.http.http_server"
SERVER_STOP_CMD = f'pkill -f "{SERVER_START_CMD}"'
# 2>&1 redirects stderr to stdout
SERVER_LOGFILE = os.path.expanduser("~/.rh/server.log")
START_SCREEN_CMD = (
    f"screen -dm bash -c \"{SERVER_START_CMD} 2>&1 | tee -a '{SERVER_LOGFILE}' 2>&1\""
)
START_NOHUP_CMD = f"nohup {SERVER_START_CMD} >> {SERVER_LOGFILE} 2>&1 &"
# We need to specify "--disable-usage-stats" because cluster.run uses INTERACTIVE (-t) SshMode by
# default, which Ray detects and asks the user for 10 seconds whether they want to opt out of usage
# stats collection. This breaks the daemon start sequence, so we disable it upfront.
RAY_START_CMD = f"ray start --head --port {DEFAULT_RAY_PORT} --disable-usage-stats"
# RAY_BOOTSTRAP_FILE = "~/ray_bootstrap_config.yaml"
# --autoscaling-config=~/ray_bootstrap_config.yaml
# We need to use this instead of ray stop to make sure we don't stop the SkyPilot ray server,
# which runs on other ports but is required to preserve autostop and correct cluster status.
RAY_KILL_CMD = 'pkill -f ".*ray.*' + str(DEFAULT_RAY_PORT) + '.*"'

CONDA_INSTALL_CMDS = [
    "wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh",
    "bash ~/miniconda.sh -b -p ~/miniconda",
    "source $HOME/miniconda/bin/activate",
]
# TODO should default to user's local Python version?
# from platform import python_version; python_version()
CONDA_PREFERRED_PYTHON_VERSION = "3.10.9"

TEST_ORG = "test-org"
TESTING_LOG_LEVEL = "INFO"

EMPTY_DEFAULT_ENV_NAME = "_cluster_default_env"
DEFAULT_DOCKER_CONTAINER_NAME = "sky_container"
DOCKER_LOGIN_ENV_VARS = {
    "SKYPILOT_DOCKER_USERNAME",
    "SKYPILOT_DOCKER_PASSWORD",
    "SKYPILOT_DOCKER_SERVER",
}

# Constants for the status check
DOUBLE_SPACE_UNICODE = "\u00A0\u00A0"
BULLET_UNICODE = "\u2022"
SECOND = 1
MINUTE = 60
HOUR = 3600
DEFAULT_STATUS_CHECK_INTERVAL = 1 * MINUTE
INCREASED_STATUS_CHECK_INTERVAL = 1 * HOUR
GPU_COLLECTION_INTERVAL = 5 * SECOND

# We collect gpu every GPU_COLLECTION_INTERVAL.
# Meaning that in one minute we collect (MINUTE / GPU_COLLECTION_INTERVAL) gpu stats.
# Currently, we save gpu info of the last 10 minutes or less.
MAX_GPU_INFO_LEN = (MINUTE / GPU_COLLECTION_INTERVAL) * 10

# If we just collect the gpu stats (and not send them to den), the gpu_info dictionary *will not* be reseted by the servlets.
# Therefore, we need to cut the gpu_info size, so it doesn't consume too much cluster memory.
# Currently, we reduce the size by half, meaning we only keep the gpu_info of the last (MAX_GPU_INFO_LEN / 2) minutes.
REDUCED_GPU_INFO_LEN = MAX_GPU_INFO_LEN / 2


# Constants Surfacing Logs to Den
DEFAULT_LOG_SURFACING_INTERVAL = 2 * MINUTE
SERVER_LOGS_FILE_NAME = "server.log"
DEFAULT_SURFACED_LOG_LENGTH = 20
# Constants for schedulers
INCREASED_INTERVAL = 1 * HOUR
