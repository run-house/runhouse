import shlex
import subprocess

from runhouse.logger import get_logger

logger = get_logger(__name__)


class AutostopHelper:
    """A helper class strictly to run SkyPilot methods on OnDemandClusters inside SkyPilot's conda env."""

    SKY_VENV = "~/skypilot-runtime"

    def __init__(self):
        self._activity_registered = False

    async def set_last_active_time_to_now(self):
        self._activity_registered = True

    def _run_python_in_sky_venv(self, cmd: str):
        sky_python_cmd = f"{self.SKY_VENV}/bin/python -c {cmd}"

        logger.debug(f"Running command in SkyPilot's venv: {sky_python_cmd}")
        # run with subprocess and return the output
        return subprocess.run(
            sky_python_cmd,
            shell=True,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        ).stdout.decode("utf-8")

    async def get_autostop(self):
        sky_get_autostop_cmd = shlex.quote(
            "from sky.skylet.autostop_lib import get_autostop_config; "
            "print(get_autostop_config().autostop_idle_minutes)"
        )

        return int(self._run_python_in_sky_venv(sky_get_autostop_cmd))

    async def get_last_active_time(self):
        sky_get_last_active_time_cmd = shlex.quote(
            "from sky.skylet.autostop_lib import get_last_active_time; "
            "print(get_last_active_time())"
        )

        return float(self._run_python_in_sky_venv(sky_get_last_active_time_cmd))

    async def set_autostop(self, idle_minutes: int):
        # Filling in "cloudvmray" as the backend because it's the only backend supported by SkyPilot right now,
        # if needed we can grab the backend from the autostop config with:
        # `from sky.skylet.autostop_lib import get_autostop_config; get_autostop_config().backend`
        sky_set_autostop_cmd = shlex.quote(
            f"from sky.skylet.autostop_lib import set_autostop; "
            f'set_autostop({idle_minutes}, "cloudvmray", True)'
        )

        self._run_python_in_sky_venv(sky_set_autostop_cmd)

    async def register_activity_if_needed(self):
        sky_register_activity_cmd = shlex.quote(
            "from sky.skylet.autostop_lib import set_last_active_time_to_now; "
            "set_last_active_time_to_now()"
        )

        if self._activity_registered:
            logger.debug("Activity registered, updating last active time in SkyConfig")
            self._run_python_in_sky_venv(sky_register_activity_cmd)
            self._activity_registered = False
        else:
            logger.debug(
                "No activity registered, not updating last active time in SkyConfig"
            )
