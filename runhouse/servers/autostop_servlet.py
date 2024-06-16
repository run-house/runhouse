import logging
import subprocess

logger = logging.getLogger(__name__)


class AutostopServlet:
    """A helper class strictly to run SkyPilot methods on OnDemandClusters inside SkyPilot's conda env."""

    def __init__(self):
        self._activity_registered = False

    def set_last_active_time_to_now(self):
        self._activity_registered = True

    def update_autostop_in_sky_config(self):
        SKY_VENV = "~/skypilot-runtime"
        SKY_AUTOSTOP_CMD = "from sky.skylet.autostop_lib import set_last_active_time_to_now; set_last_active_time_to_now()"
        SKY_CMD = f"{SKY_VENV}/bin/python -c '{SKY_AUTOSTOP_CMD}'"

        if self._activity_registered:
            logger.debug(
                "Activity registered, updating last active time in SkyConfig with command: {SKY_CMD}"
            )
            subprocess.run(SKY_CMD, shell=True, check=True)
            self._activity_registered = False
        else:
            logger.debug(
                "No activity registered, not updating last active time in SkyConfig"
            )
