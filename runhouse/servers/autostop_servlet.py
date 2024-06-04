import time


class AutostopServlet:
    """A helper class strictly to run SkyPilot methods on OnDemandClusters inside SkyPilot's conda env."""

    def __init__(self):
        self._last_activity = time.time()
        self._last_register = None

    def set_last_active_time_to_now(self):
        self._last_activity = time.time()

    def set_autostop(self, value=None):
        from sky.skylet import autostop_lib

        self.set_last_active_time_to_now()
        autostop_lib.set_autostop(value, None, True)

    def update_autostop_in_sky_config(self):
        import pickle

        from sky.skylet import configs as sky_configs

        autostop_mins = pickle.loads(
            sky_configs.get_config("autostop_config")
        ).autostop_idle_minutes
        if autostop_mins > 0 and (
            self._last_register is None or self._last_register < self._last_activity
        ):
            sky_configs.set_config("autostop_last_active_time", self._last_activity)
            self._last_register = self._last_activity
