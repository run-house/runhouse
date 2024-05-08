import os
import sys
from pathlib import Path
from typing import List

RESERVED_SYSTEM_NAMES: List[str] = ["file", "s3", "gs", "azure", "here", "ssh", "sftp"]
CLUSTER_CONFIG_PATH: str = "~/.rh/cluster_config.json"
LOCALHOST: str = "127.0.0.1"
LOCAL_HOSTS: List[str] = ["localhost", LOCALHOST]

LOGS_DIR = ".rh/logs"
RH_LOGFILE_PATH = Path.home() / LOGS_DIR

ENVS_DIR = "~/.rh/envs"

MAX_MESSAGE_LENGTH = 1 * 1024 * 1024 * 1024  # 1 GB

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
    "source $HOME/miniconda3/bin/activate",
]

TEST_ORG = "test-org"

EMPTY_DEFAULT_ENV_NAME = "_cluster_default_env"
