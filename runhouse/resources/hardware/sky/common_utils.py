# Source: https://github.com/skypilot-org/skypilot/blob/feb52cf/sky/utils/common_utils.py

import getpass
import hashlib
import os
import random
import socket
from typing import Optional
import uuid

from runhouse.resources.hardware.sky import constants


USER_HASH_LENGTH = 8
_USER_HASH_FILE = os.path.expanduser('~/.sky/user_hash')


def user_and_hostname_hash() -> str:
    """Returns a string containing <user>-<hostname hash last 4 chars>.

    For uniquefying user clusters on shared-account cloud providers. Also used
    for AWS security group.

    Using uuid.getnode() instead of gethostname() is incorrect; observed to
    collide on Macs.

    NOTE: BACKWARD INCOMPATIBILITY NOTES

    Changing this string will render AWS clusters shown in `sky status`
    unreusable and potentially cause leakage:

    - If a cluster is STOPPED, any command restarting it (`sky launch`, `sky
      start`) will launch a NEW cluster.
    - If a cluster is UP, a `sky launch` command reusing it will launch a NEW
      cluster. The original cluster will be stopped and thus leaked from Sky's
      perspective.
    - `sky down/stop/exec` on these pre-change clusters still works, if no new
      clusters with the same name have been launched.

    The reason is AWS security group names are derived from this string, and
    thus changing the SG name makes these clusters unrecognizable.
    """
    hostname_hash = hashlib.md5(socket.gethostname().encode()).hexdigest()[-4:]
    return f'{getpass.getuser()}-{hostname_hash}'


def get_user_hash() -> str:
    """Returns a unique user-machine specific hash as a user id.

    We cache the user hash in a file to avoid potential user_name or
    hostname changes causing a new user hash to be generated.
    """

    def _is_valid_user_hash(user_hash: Optional[str]) -> bool:
        if user_hash is None:
            return False
        try:
            int(user_hash, 16)
        except (TypeError, ValueError):
            return False
        return len(user_hash) == USER_HASH_LENGTH

    user_hash = os.getenv(constants.USER_ID_ENV_VAR)
    if _is_valid_user_hash(user_hash):
        assert user_hash is not None
        return user_hash

    if os.path.exists(_USER_HASH_FILE):
        # Read from cached user hash file.
        with open(_USER_HASH_FILE, 'r') as f:
            # Remove invalid characters.
            user_hash = f.read().strip()
        if _is_valid_user_hash(user_hash):
            return user_hash

    hash_str = user_and_hostname_hash()
    user_hash = hashlib.md5(hash_str.encode()).hexdigest()[:USER_HASH_LENGTH]
    if not _is_valid_user_hash(user_hash):
        # A fallback in case the hash is invalid.
        user_hash = uuid.uuid4().hex[:USER_HASH_LENGTH]
    os.makedirs(os.path.dirname(_USER_HASH_FILE), exist_ok=True)
    with open(_USER_HASH_FILE, 'w') as f:
        f.write(user_hash)
    return user_hash

class Backoff:
    """Exponential backoff with jittering."""
    MULTIPLIER = 1.6
    JITTER = 0.4

    def __init__(self, initial_backoff: int = 5, max_backoff_factor: int = 5):
        self._initial = True
        self._backoff = 0.0
        self._inital_backoff = initial_backoff
        self._max_backoff = max_backoff_factor * self._inital_backoff

    # https://github.com/grpc/grpc/blob/2d4f3c56001cd1e1f85734b2f7c5ce5f2797c38a/doc/connection-backoff.md
    # https://github.com/grpc/grpc/blob/5fc3ff82032d0ebc4bf252a170ebe66aacf9ed9d/src/core/lib/backoff/backoff.cc

    def current_backoff(self) -> float:
        """Backs off once and returns the current backoff in seconds."""
        if self._initial:
            self._initial = False
            self._backoff = min(self._inital_backoff, self._max_backoff)
        else:
            self._backoff = min(self._backoff * self.MULTIPLIER,
                                self._max_backoff)
        self._backoff += random.uniform(-self.JITTER * self._backoff,
                                        self.JITTER * self._backoff)
        return self._backoff
