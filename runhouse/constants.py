from pathlib import Path
from typing import List

RESERVED_SYSTEM_NAMES: List[str] = ["file", "s3", "gs", "azure", "here", "ssh", "sftp"]
CLUSTER_CONFIG_PATH: str = "~/.rh/cluster_config.json"
LOCALHOST: str = "127.0.0.1"

LOGS_DIR = ".rh/logs"
RH_LOGFILE_PATH = Path.home() / LOGS_DIR

MAX_MESSAGE_LENGTH = 1 * 1024 * 1024 * 1024  # 1 GB
