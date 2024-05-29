import asyncio
import copy
import json
import logging
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union

import requests
from pydantic import BaseModel

import runhouse

from runhouse.constants import (
    CLUSTER_CONFIG_PATH,
    DEFAULT_LOG_SEND_INTERVAL,
    DEFAULT_STATUS_CHECK_INTERVAL,
    DEFAULT_STATUS_LOG_LENGTH,
    INCREASED_INTERVAL,
    SCHEDULERS_DELAY,
    SERVER_LOGFILE,
)

from runhouse.globals import configs, obj_store, rns_client
from runhouse.resources.hardware import load_cluster_config_from_file
from runhouse.rns.utils.api import ResourceAccess
from runhouse.servers.http.auth import AuthCache

from runhouse.utils import sync_function

logger = logging.getLogger(__name__)


class ClusterServletError(Exception):
    pass


# This is a copy of the Pydantic model that we use to validate in Den
class ResourceStatusData(BaseModel):
    cluster_config: dict
    env_resource_mapping: Dict[str, List[Dict[str, Any]]]
    system_cpu_usage: float
    system_memory_usage: Dict[str, Any]
    system_disk_usage: Dict[str, Any]
    env_servlet_processes: Dict[str, Dict[str, Any]]
    server_pid: int
    runhouse_version: str


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

        if cluster_config.get("resource_subtype", None) == "OnDemandCluster":
            if cluster_config.get("autostop_mins") > 0:
                try:
                    from sky.skylet import configs as sky_configs  # noqa
                except ImportError:
                    raise ImportError(
                        "skypilot must be installed on the cluster environment to support cluster autostop. "
                        "Install using cluster.run('pip install skypilot') or adding `skypilot` to the env requirements."
                    )
            self._last_activity = time.time()
            self._last_register = None
            autostop_thread = threading.Thread(target=self.update_autostop, daemon=True)
            autostop_thread.start()

        # Only send for clusters that have den_auth enabled and if we are logged in with a user's token
        # to authenticate the request
        if self.cluster_config.get("den_auth", False) and configs.token:
            logger.debug("Creating send_status_info_to_den thread.")
            post_status_thread = threading.Thread(
                target=self.send_status_info_to_den, daemon=True
            )
            post_status_thread.start()

            logger.debug("Creating send_logs_to_den thread.")
            send_logs_thread = threading.Thread(
                target=self.send_cluster_logs_to_den, daemon=True
            )
            send_logs_thread.start()

    ##############################################
    # Cluster autostop
    ##############################################
    def update_autostop(self):
        import pickle

        from sky.skylet import configs as sky_configs

        while True:
            autostop_mins = pickle.loads(
                sky_configs.get_config("autostop_config")
            ).autostop_idle_minutes
            self._last_register = float(
                sky_configs.get_config("autostop_last_active_time")
            )
            if autostop_mins > 0 and (
                not self._last_register
                or (
                    # within 2 min of autostop and there's more recent activity
                    60 * autostop_mins - (time.time() - self._last_register) < 120
                    and self._last_activity > self._last_register
                )
            ):
                sky_configs.set_config("autostop_last_active_time", self._last_activity)
                self._last_register = self._last_activity

            time.sleep(30)

    ##############################################
    # Cluster config state storage methods
    ##############################################
    async def aget_cluster_config(self) -> Dict[str, Any]:
        return self.cluster_config

    async def aupdate_status_check_interval_in_cluster_config(self):
        cluster_path = Path(CLUSTER_CONFIG_PATH).expanduser()
        with open(cluster_path) as cluster_local_config:
            local_config = json.load(cluster_local_config)
            local_interval = local_config.get("status_check_interval")
            servlet_interval = self.cluster_config.get("status_check_interval")
            if (
                servlet_interval
                and local_interval
                and local_interval != servlet_interval
            ):
                await self.aset_cluster_config(local_config)
                if local_interval > 0:
                    logger.info(
                        f"Updated cluster_config with new status check interval: {round(local_interval/60, 2)} minutes."
                    )
                return True
            else:
                return False

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
        if key == "autostop_mins" and value > -1:
            from sky.skylet import configs as sky_configs

            self._last_activity = time.time()
            sky_configs.set_config("autostop_last_active_time", self._last_activity)
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
        self._last_activity = time.time()
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

    async def asend_status_info_to_den(self):
        # Delay the start of post_status_thread, so we'll finish the cluster startup properly
        await asyncio.sleep(SCHEDULERS_DELAY)
        while True:
            logger.info("Trying to send cluster status to Den.")
            try:
                is_config_updated = (
                    await self.aupdate_status_check_interval_in_cluster_config()
                )
                interval_size = (await self.aget_cluster_config()).get(
                    "status_check_interval", DEFAULT_STATUS_CHECK_INTERVAL
                )
                den_auth = (await self.aget_cluster_config()).get(
                    "den_auth", DEFAULT_STATUS_CHECK_INTERVAL
                )
                if interval_size == -1:
                    if is_config_updated:
                        logger.info(
                            f"Disabled periodic cluster status check. For enabling it, please run "
                            f"cluster.restart_server(). If you want to set up an interval size that is not the "
                            f"default value {round(DEFAULT_STATUS_CHECK_INTERVAL/60,2)} please run "
                            f"cluster._enable_or_update_status_check(interval_size) after restarting the server."
                        )
                    break
                if not den_auth:
                    logger.info(
                        f"Disabled periodic cluster status check because den_auth is disabled. For enabling it, please run "
                        f"cluster.restart_server() and make sure that den_auth is enabled. If you want to set up an interval size that is not the "
                        f"default value {round(DEFAULT_STATUS_CHECK_INTERVAL / 60, 2)} please run "
                        f"cluster._enable_or_update_status_check(interval_size) after restarting the server."
                    )
                    break
                status: ResourceStatusData = await self.astatus()
                status_data = {
                    "status": "running",
                    "resource_type": status.cluster_config.get("resource_type"),
                    "data": dict(status),
                }
                cluster_uri = rns_client.format_rns_address(
                    (await self.aget_cluster_config()).get("name")
                )
                api_server_url = status.cluster_config.get(
                    "api_server_url", rns_client.api_server_url
                )
                post_status_data_resp = requests.post(
                    f"{api_server_url}/resource/{cluster_uri}/cluster/status",
                    data=json.dumps(status_data),
                    headers=rns_client.request_headers(),
                )
                if post_status_data_resp.status_code != 200:
                    logger.error(
                        f"({post_status_data_resp.status_code}) Failed to send cluster status check to Den: {post_status_data_resp.text}"
                    )
                else:
                    logger.info(
                        f"Successfully updated cluster status in Den. Next status check will be in {round(interval_size / 60, 2)} minutes."
                    )
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
                await asyncio.sleep(INCREASED_INTERVAL)
            finally:
                await asyncio.sleep(interval_size)

    def send_status_info_to_den(self):
        asyncio.run(self.asend_status_info_to_den())

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

    async def asend_cluster_logs_to_den(
        self, num_of_lines: int = DEFAULT_STATUS_LOG_LENGTH
    ):
        # Delay the start of post_logs_thread, so we'll finish the cluster startup properly
        await asyncio.sleep(SCHEDULERS_DELAY)

        while True:
            logger.info("Trying to send cluster logs to Den")
            try:
                interval_size = DEFAULT_LOG_SEND_INTERVAL
                latest_logs = self._get_logs(num_of_lines=num_of_lines)
                s3_file_name = ""
                logs_data = {"file_name": s3_file_name, "logs": latest_logs}
                cluster_config = await self.aget_cluster_config()
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
