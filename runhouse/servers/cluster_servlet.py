import asyncio
import copy
import json
import logging
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union

import requests

import runhouse

from runhouse.constants import (
    CLUSTER_CONFIG_PATH,
    DEFAULT_LOG_SURFACING_INTERVAL,
    DEFAULT_STATUS_CHECK_INTERVAL,
    DEFAULT_SURFACED_LOG_LENGTH,
    INCREASED_INTERVAL,
    INCREASED_STATUS_CHECK_INTERVAL,
    S3_LOGS_FILE_NAME,
    SCHEDULERS_DELAY,
    SERVER_LOGFILE,
)

from runhouse.globals import configs, obj_store, rns_client
from runhouse.resources.hardware import load_cluster_config_from_file
from runhouse.rns.rns_client import ResourceStatusData
from runhouse.rns.utils.api import ResourceAccess
from runhouse.servers.autostop_servlet import AutostopServlet
from runhouse.servers.http.auth import AuthCache

from runhouse.utils import sync_function

logger = logging.getLogger(__name__)


class ClusterServletError(Exception):
    pass


class ClusterServlet:
    async def __init__(
        self, cluster_config: Optional[Dict[str, Any]] = None, *args, **kwargs
    ):

        # We do this here instead of at the start of the HTTP Server startup
        # because someone can be running `HTTPServer()` standalone in a test
        # and still want an initialized cluster config in the servlet.
        if not cluster_config:
            cluster_config = load_cluster_config_from_file()

        self.cluster_config: Optional[Dict[str, Any]] = (
            cluster_config if cluster_config else {}
        )
        self._initialized_env_servlet_names: Set[str] = set()
        self._key_to_env_servlet_name: Dict[Any, str] = {}
        self._auth_cache: AuthCache = AuthCache(cluster_config)
        self.autostop_servlet = None

        if cluster_config.get("resource_subtype", None) == "OnDemandCluster":
            import ray

            current_ip = ray.get_runtime_context().worker.node_ip_address
            self.autostop_servlet = (
                ray.remote(AutostopServlet)
                .options(
                    name="autostop_servlet",
                    get_if_exists=True,
                    lifetime="detached",
                    namespace="runhouse",
                    max_concurrency=1000,
                    resources={f"node:{current_ip}": 0.001},
                    num_cpus=0,
                    runtime_env={"env_vars": {"VIRTUAL_ENV": "skypilot-runtime"}},
                )
                .remote()
            )

        # Only send for clusters that have den_auth enabled and if we are logged in with a user's token
        # to authenticate the request
        if self.cluster_config.get("den_auth", False):
            logger.info("Creating periodic_status_check thread.")
            post_status_thread = threading.Thread(
                target=self.periodic_status_check, daemon=True
            )
            post_status_thread.start()

            logger.debug("Creating send_logs_to_den thread.")
            send_logs_thread = threading.Thread(
                target=self.send_cluster_logs_to_den, daemon=True
            )
            send_logs_thread.start()

    ##############################################
    # Cluster config state storage methods
    ##############################################
    async def aget_cluster_config(self) -> Dict[str, Any]:
        return self.cluster_config

    async def async_local_and_servlet_cluster_configs(self, values_to_sync: list[str]):
        cluster_path = Path(CLUSTER_CONFIG_PATH).expanduser()
        with open(cluster_path) as cluster_local_config:
            local_config = json.load(cluster_local_config)
            is_synced = False
            for item in values_to_sync:
                local_cluster_value = local_config.get(item)
                servlet_cluster_value = self.cluster_config.get(item)
                if (
                    local_cluster_value
                    and servlet_cluster_value
                    and local_cluster_value != servlet_cluster_value
                ):
                    if item == "status_check_interval" and servlet_cluster_value > 0:
                        logger.info(
                            f"Updated cluster_config with new status check interval: {round(servlet_cluster_value/60, 2)} minutes."
                        )
                    if item == "logs_surfacing_interval" and servlet_cluster_value > 0:
                        logger.info(
                            f"Updated cluster_config with new logs surfacing interval: {round(servlet_cluster_value/60, 2)} minutes."
                        )
                    is_synced = True
            if is_synced:
                await self.aset_cluster_config(local_config)
            return is_synced

    async def aset_cluster_config(self, cluster_config: Dict[str, Any]):
        self.cluster_config = cluster_config

        # Propagate the changes to all other process's obj_stores
        await asyncio.gather(
            *[
                obj_store.acall_env_servlet_method(
                    env_servlet_name,
                    "aset_cluster_config",
                    cluster_config,
                    use_env_servlet_cache=False,
                )
                for env_servlet_name in await self.aget_all_initialized_env_servlet_names()
            ]
        )

        return self.cluster_config

    async def aset_cluster_config_value(self, key: str, value: Any):
        if self.autostop_servlet and key == "autostop_mins" and value > -1:
            await self.autostop_servlet.set_auto_stop.remote(value)
        self.cluster_config[key] = value

        # Propagate the changes to all other process's obj_stores
        await asyncio.gather(
            *[
                obj_store.acall_env_servlet_method(
                    env_servlet_name,
                    "aset_cluster_config_value",
                    key,
                    value,
                    use_env_servlet_cache=False,
                )
                for env_servlet_name in await self.aget_all_initialized_env_servlet_names()
            ]
        )

        return self.cluster_config

    ##############################################
    # Auth cache internal functions
    ##############################################
    async def aresource_access_level(
        self, token: str, resource_uri: str
    ) -> Union[str, None]:
        # If the token in this request matches that of the owner of the cluster,
        # they have access to everything
        if configs.token and (
            configs.token == token
            or rns_client.cluster_token(configs.token, resource_uri) == token
        ):
            return ResourceAccess.WRITE
        return self._auth_cache.lookup_access_level(token, resource_uri)

    async def aget_username(self, token: str) -> str:
        return self._auth_cache.get_username(token)

    async def ahas_resource_access(self, token: str, resource_uri=None) -> bool:
        """Checks whether user has read or write access to a given module saved on the cluster."""
        from runhouse.rns.utils.api import ResourceAccess

        if token is None:
            # If no token is provided assume no access
            return False

        cluster_uri = self.cluster_config["name"]
        cluster_access = await self.aresource_access_level(token, cluster_uri)
        if cluster_access == ResourceAccess.WRITE:
            # if user has write access to cluster will have access to all resources
            return True

        if resource_uri is None and cluster_access not in [
            ResourceAccess.WRITE,
            ResourceAccess.READ,
        ]:
            # If module does not have a name, must have access to the cluster
            return False

        resource_access_level = await self.aresource_access_level(token, resource_uri)
        if resource_access_level not in [ResourceAccess.WRITE, ResourceAccess.READ]:
            return False

        return True

    async def aclear_auth_cache(self, token: str = None):
        self._auth_cache.clear_cache(token)

    ##############################################
    # Key to servlet where it is stored mapping
    ##############################################
    async def amark_env_servlet_name_as_initialized(self, env_servlet_name: str):
        self._initialized_env_servlet_names.add(env_servlet_name)

    async def ais_env_servlet_name_initialized(self, env_servlet_name: str) -> bool:
        return env_servlet_name in self._initialized_env_servlet_names

    async def aget_all_initialized_env_servlet_names(self) -> Set[str]:
        return self._initialized_env_servlet_names

    async def aget_key_to_env_servlet_name_dict_keys(self) -> List[Any]:
        return list(self._key_to_env_servlet_name.keys())

    async def aget_key_to_env_servlet_name_dict(self) -> Dict[Any, str]:
        return self._key_to_env_servlet_name

    async def aget_env_servlet_name_for_key(self, key: Any) -> str:
        if self.autostop_servlet:
            await self.autostop_servlet.set_last_active_time_to_now.remote()
        return self._key_to_env_servlet_name.get(key, None)

    async def aput_env_servlet_name_for_key(self, key: Any, env_servlet_name: str):
        if not await self.ais_env_servlet_name_initialized(env_servlet_name):
            raise ValueError(
                f"Env servlet name {env_servlet_name} not initialized, and you tried to mark a resource as in it."
            )
        self._key_to_env_servlet_name[key] = env_servlet_name

    async def apop_env_servlet_name_for_key(self, key: Any, *args) -> str:
        # *args allows us to pass default or not
        return self._key_to_env_servlet_name.pop(key, *args)

    async def aclear_key_to_env_servlet_name_dict(self):
        self._key_to_env_servlet_name = {}

    ##############################################
    # Remove Env Servlet
    ##############################################
    async def aclear_all_references_to_env_servlet_name(self, env_servlet_name: str):
        self._initialized_env_servlet_names.remove(env_servlet_name)
        deleted_keys = [
            key
            for key, env in self._key_to_env_servlet_name.items()
            if env == env_servlet_name
        ]
        for key in deleted_keys:
            self._key_to_env_servlet_name.pop(key)
        return deleted_keys

    ##############################################
    # Cluster status functions
    ##############################################

    async def aperiodic_status_check(self):
        # Delay the start of post_status_thread, so we'll finish the cluster startup properly
        await asyncio.sleep(SCHEDULERS_DELAY)
        while True:
            try:
                await self.aupdate_status_check_interval_in_cluster_config()

                cluster_config = await self.aget_cluster_config()
                interval_size = cluster_config.get(
                    "status_check_interval", DEFAULT_STATUS_CHECK_INTERVAL
                )
                den_auth = cluster_config.get("den_auth", False)

                # Only if one of these is true, do we actually need to get the status from each EnvServlet
                should_send_status_to_den = den_auth and interval_size != -1
                should_update_autostop = self.autostop_servlet is not None
                if should_send_status_to_den or should_update_autostop:
                    logger.info(
                        "Performing cluster status check: potentially sending to Den or updating autostop."
                    )
                    status: ResourceStatusData = await self.astatus()

                    if should_update_autostop:
                        function_running = any(
                            any(
                                len(resource_info["active_function_calls"]) > 0
                                for resource_info in resources
                            )
                            for resources in status.env_resource_mapping.values()
                        )
                        if function_running:
                            await self.autostop_servlet.set_last_active_time_to_now.remote()
                        await self.autostop_servlet.update_autostop_in_sky_config.remote()

                    if should_send_status_to_den:
                        cluster_rns_address = cluster_config.get("name")
                        await rns_client.send_status(status, cluster_rns_address)
            except Exception as e:
                logger.error(
                    f"Cluster status check has failed: {e}. Please check cluster logs for more info."
                )
                logger.warning(
                    f"Temporarily increasing the interval between two consecutive status checks. "
                    f"Next status check will be in {round(INCREASED_INTERVAL / 60, 2)} minutes. "
                    f"For changing the interval size, please run cluster._enable_or_update_status_check(new_interval). "
                    f"If a value is not provided, interval size will be set to {DEFAULT_STATUS_CHECK_INTERVAL}"
                )
                await asyncio.sleep(INCREASED_STATUS_CHECK_INTERVAL)
            else:

                await asyncio.sleep(interval_size)

    def periodic_status_check(self):
        # This is only ever called once in its own thread, so we can do asyncio.run here instead of
        # sync_function.
        asyncio.run(self.aperiodic_status_check())

    async def _status_for_env_servlet(self, env_servlet_name):
        try:
            (
                objects_in_env_servlet,
                env_servlet_utilization_data,
            ) = await obj_store.acall_env_servlet_method(
                env_servlet_name, method="astatus_local"
            )

            return {
                "env_servlet_name": env_servlet_name,
                "objects_in_env_servlet": objects_in_env_servlet,
                "env_servlet_utilization_data": env_servlet_utilization_data,
            }

        # Need to catch the exception here because we're running this in a gather,
        # and need to know which env servlet failed
        except Exception as e:
            return {"env_servlet_name": env_servlet_name, "Exception": e}

    async def astatus(self):
        import psutil

        from runhouse.utils import get_pid

        config_cluster = copy.deepcopy(self.cluster_config)

        # Popping out creds because we don't want to show them in the status
        config_cluster.pop("creds", None)

        # Getting data from each env servlet about the objects it contains and the utilization data
        env_resource_mapping = {}
        env_servlet_utilization_data = {}
        env_servlets_status = await asyncio.gather(
            *[
                self._status_for_env_servlet(env_servlet_name)
                for env_servlet_name in self._initialized_env_servlet_names
            ],
        )

        # Store the data for the appropriate env servlet name
        for env_status in env_servlets_status:
            env_servlet_name = env_status.get("env_servlet_name")

            # Nothing if there was an exception
            if "Exception" in env_status.keys():
                e = env_status.get("Exception")
                logger.warning(
                    f"Exception {str(e)} in status for env servlet {env_servlet_name}"
                )
                env_resource_mapping[env_servlet_name] = []
                env_servlet_utilization_data[env_servlet_name] = {}

            # Otherwise, store what was in the env and the utilization data
            else:
                env_resource_mapping[env_servlet_name] = env_status.get(
                    "objects_in_env_servlet"
                )
                env_servlet_utilization_data[env_servlet_name] = env_status.get(
                    "env_servlet_utilization_data"
                )

        # TODO: decide if we need this info at all: cpu_usage, memory_usage, disk_usage
        cpu_usage = psutil.cpu_percent(interval=1)

        # Fields: `available`, `percent`, `used`, `free`, `active`, `inactive`, `buffers`, `cached`, `shared`, `slab`
        memory_usage = psutil.virtual_memory()._asdict()

        # Fields: `total`, `used`, `free`, `percent`
        disk_usage = psutil.disk_usage("/")._asdict()

        status_data = {
            "cluster_config": config_cluster,
            "runhouse_version": runhouse.__version__,
            "server_pid": get_pid(),
            "env_resource_mapping": env_resource_mapping,
            "env_servlet_processes": env_servlet_utilization_data,
            "system_cpu_usage": cpu_usage,
            "system_memory_usage": memory_usage,
            "system_disk_usage": disk_usage,
        }
        status_data = ResourceStatusData(**status_data)
        return status_data

    def status(self):
        return sync_function(self.astatus)()

    ##############################################
    # Surface cluster logs to Den
    ##############################################
    def _get_logs(self, num_of_lines: int):
        with open(SERVER_LOGFILE) as log_file:
            log_lines = log_file.readlines()
            if num_of_lines >= len(log_lines):
                return " ".join(log_lines)
            return " ".join(log_lines[-num_of_lines:])

    async def asend_cluster_logs_to_den(self):
        # Delay the start of post_logs_thread, so we'll finish the cluster startup properly
        await asyncio.sleep(SCHEDULERS_DELAY)

        while True:
            logger.info("Trying to send cluster logs to Den")
            try:
                interval_size = DEFAULT_LOG_SURFACING_INTERVAL

                cluster_config = await self.aget_cluster_config()

                is_config_updated = await self.async_local_and_servlet_cluster_configs(
                    ["logs_surfacing_interval", "surfaced_logs_length"]
                )

                interval_size = (await self.aget_cluster_config()).get(
                    "logs_surfacing_interval", DEFAULT_STATUS_CHECK_INTERVAL
                )

                num_of_lines = (await self.aget_cluster_config()).get(
                    "surfaced_logs_length", DEFAULT_STATUS_CHECK_INTERVAL
                )

                den_auth = (await self.aget_cluster_config()).get("den_auth")

                # check if the scheduler needs to stop running.
                if interval_size == -1 or num_of_lines == 0:
                    if is_config_updated:
                        logger.info(
                            f"Disabled cluster logs surfacing. For enabling it, please run "
                            f"cluster.restart_server()\n. If you want to set the interval size and/or the log tail "
                            f"length to values that are not the default ones "
                            f"({round(DEFAULT_LOG_SURFACING_INTERVAL / 60, 2)} minutes, {DEFAULT_SURFACED_LOG_LENGTH} "
                            f"lines), please run cluster._enable_or_update_log_surface_to_den(num_of_lines, interval_size) "
                            f"after restarting the server."
                        )
                    break
                if not den_auth:
                    logger.info(
                        f"Disabled cluster logs surfacing because den_auth is disabled. For enabling it, please run "
                        f"cluster.restart_server() and make sure that den_auth is enabled. If you want to set up an interval size that is not the "
                        f"default value {round(DEFAULT_LOG_SURFACING_INTERVAL / 60, 2)}, or save more logs lines than {DEFAULT_SURFACED_LOG_LENGTH}, please run "
                        f"cluster._enable_or_update_log_surface_to_den(num_of_lines, interval_size) after restarting the server."
                    )
                    break

                latest_logs = self._get_logs(num_of_lines=num_of_lines)
                logs_data = {"file_name": S3_LOGS_FILE_NAME, "logs": latest_logs}

                cluster_uri = rns_client.format_rns_address(cluster_config.get("name"))
                api_server_url = cluster_config.get(
                    "api_server_url", rns_client.api_server_url
                )

                post_logs_resp = requests.post(
                    f"{api_server_url}/resource/{cluster_uri}/logs",
                    data=json.dumps(logs_data),
                    headers=rns_client.request_headers(),
                )

                if post_logs_resp.status_code != 200:
                    logger.error(
                        f"({post_logs_resp.status_code}) Failed to send cluster logs to Den: {post_logs_resp.text}"
                    )
                else:
                    logger.info(
                        f"Successfully sent cluster logs to Den. Next status check will be in {round(interval_size / 60, 2)} minutes."
                    )
            except Exception as e:
                logger.error(
                    f"Sending cluster logs to den has failed: {e}. Please check cluster logs for more info."
                )
                logger.warning(
                    f"Temporarily increasing the interval between two consecutive log retrievals."
                    f"Next log retrieval will be in {round(INCREASED_INTERVAL / 60, 2)} minutes. "
                    f"For changing the interval size, please run cluster.restart_server(). "
                    f"Interval size will be set to {interval_size}"
                )
            finally:
                await asyncio.sleep(interval_size)

    def send_cluster_logs_to_den(self):
        asyncio.run(self.asend_cluster_logs_to_den())
