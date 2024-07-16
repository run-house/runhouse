import asyncio
import copy
import json
import threading
from typing import Any, Dict, List, Optional, Set, Union

import requests

import runhouse

from runhouse.constants import (
    DEFAULT_LOG_LEVEL,
    DEFAULT_LOG_SURFACING_INTERVAL,
    DEFAULT_STATUS_CHECK_INTERVAL,
    INCREASED_INTERVAL,
    INCREASED_STATUS_CHECK_INTERVAL,
    S3_LOGS_FILE_NAME,
    SCHEDULERS_DELAY,
    SERVER_LOGFILE,
)

from runhouse.globals import configs, obj_store, rns_client
from runhouse.logger import ColoredFormatter, logger
from runhouse.resources.hardware import load_cluster_config_from_file
from runhouse.resources.hardware.utils import detect_cuda_version_or_cpu
from runhouse.rns.rns_client import ResourceStatusData
from runhouse.rns.utils.api import ResourceAccess
from runhouse.servers.autostop_helper import AutostopHelper
from runhouse.servers.http.auth import AuthCache

from runhouse.utils import sync_function


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
        self.autostop_helper = None

        logger.setLevel(kwargs.get("logs_level", DEFAULT_LOG_LEVEL))
        self.logger = logger

        if cluster_config.get("resource_subtype", None) == "OnDemandCluster":
            self.autostop_helper = AutostopHelper()

        logger.info("Creating periodic_cluster_checks thread.")
        cluster_checks_thread = threading.Thread(
            target=self.periodic_cluster_checks, daemon=True
        )
        cluster_checks_thread.start()

    ##############################################
    # Cluster config state storage methods
    ##############################################
    async def aget_cluster_config(self) -> Dict[str, Any]:
        return self.cluster_config

    async def aset_cluster_config(self, cluster_config: Dict[str, Any]):
        if "has_cuda" not in cluster_config.keys():
            cluster_config["has_cuda"] = detect_cuda_version_or_cpu() != "cpu"

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
        if self.autostop_helper and key == "autostop_mins":
            await self.autostop_helper.set_autostop(value)
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
        if self.autostop_helper:
            await self.autostop_helper.set_last_active_time_to_now()
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

    async def aperiodic_cluster_checks(self):
        # Delay the start of post_status_thread, so we'll finish the cluster startup properly
        await asyncio.sleep(SCHEDULERS_DELAY)
        while True:
            try:

                cluster_config = await self.aget_cluster_config()
                interval_size = cluster_config.get(
                    "status_check_interval", DEFAULT_STATUS_CHECK_INTERVAL
                )

                # Only if one of these is true, do we actually need to get the status from each EnvServlet
                should_send_status_and_logs_to_den = (
                    configs.token is not None and interval_size != -1
                )
                should_update_autostop = self.autostop_helper is not None
                if should_send_status_and_logs_to_den or should_update_autostop:
                    logger.info(
                        "Performing cluster checks: potentially sending to Den, surfacing logs to Den or updating autostop."
                    )
                    status: ResourceStatusData = await self.astatus()

                    if should_update_autostop:
                        function_running = any(
                            any(
                                len(
                                    resource["env_resource_mapping"][resource_name].get(
                                        "active_function_calls", []
                                    )
                                )
                                > 0
                                for resource_name in resource[
                                    "env_resource_mapping"
                                ].keys()
                            )
                            for resource in status.env_servlet_processes.values()
                        )
                        if function_running:
                            await self.autostop_helper.set_last_active_time_to_now()
                        # We do this separately from the set_last_active_time_to_now call above because
                        # function_running will only reflect activity from functions which happen to be running during
                        # the status check. We still need to attempt to register activity for functions which have
                        # been called and completed.
                        await self.autostop_helper.register_activity_if_needed()

                    if should_send_status_and_logs_to_den:
                        cluster_rns_address = cluster_config.get("name")
                        cluster_uri = rns_client.format_rns_address(cluster_rns_address)
                        api_server_url = status.cluster_config.get(
                            "api_server_url", rns_client.api_server_url
                        )

                        sent_status = await rns_client.send_status(
                            status=status,
                            cluster_uri=cluster_uri,
                            api_server_url=api_server_url,
                        )

                        if sent_status == 404:
                            logger.info(
                                "Cluster has not been saved, no update to status or logs."
                            )
                        elif sent_status != 200:
                            logger.error(
                                f"{sent_status}: Error in sending cluster status to Den. Check cluster logs for more info."
                            )
                        else:
                            logger.info("Successfully sent cluster status to Den.")

                            sent_logs = await rns_client.send_cluster_logs_to_den(
                                cluster_uri=cluster_uri, api_server_url=api_server_url
                            )

                            if sent_logs != 200:
                                logger.error(
                                    f"{sent_logs}: Error in sending cluster logs to Den. Check cluster logs for more info."
                                )
                            else:
                                logger.info("Successfully sent cluster logs to Den.")

            except Exception as e:
                self.logger.error(
                    f"Cluster status check has failed: {e}. Please check cluster logs for more info."
                )
                self.logger.warning(
                    f"Temporarily increasing the interval between two consecutive status checks. "
                    f"Next status check will be in {round(INCREASED_INTERVAL / 60, 2)} minutes. "
                    f"For changing the interval size, please run cluster._enable_or_update_status_check(new_interval). "
                    f"If a value is not provided, interval size will be set to {DEFAULT_STATUS_CHECK_INTERVAL}"
                )
                await asyncio.sleep(INCREASED_STATUS_CHECK_INTERVAL)
            else:
                # make sure that the thread will go to sleep, even if the interval size == -1
                # (meaning that sending status to den is disabled).
                interval_size = (
                    DEFAULT_STATUS_CHECK_INTERVAL
                    if interval_size == -1
                    else interval_size
                )
                await asyncio.sleep(interval_size)

    def periodic_cluster_checks(self):
        # This is only ever called once in its own thread, so we can do asyncio.run here instead of
        # sync_function.
        asyncio.run(self.aperiodic_cluster_checks())

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
                self.logger.warning(
                    f"Exception {str(e)} in status for env servlet {env_servlet_name}"
                )
                env_servlet_utilization_data[env_servlet_name] = {}

            # Otherwise, store what was in the env and the utilization data
            else:
                env_memory_info = env_status.get("env_servlet_utilization_data")
                env_memory_info["env_resource_mapping"] = env_status.get(
                    "objects_in_env_servlet"
                )
                env_servlet_utilization_data[env_servlet_name] = env_memory_info

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
    def _get_logs(self):
        with open(SERVER_LOGFILE) as log_file:
            log_lines = log_file.readlines()
        cleaned_log_lines = [ColoredFormatter.format_log(line) for line in log_lines]
        return " ".join(cleaned_log_lines)

    async def asend_cluster_logs_to_den(self):
        # Delay the start of post_logs_thread, so we'll finish the cluster startup properly
        await asyncio.sleep(SCHEDULERS_DELAY)

        while True:
            self.logger.info("Trying to send cluster logs to Den")
            try:
                interval_size = DEFAULT_LOG_SURFACING_INTERVAL
                latest_logs = self._get_logs()
                logs_data = {"file_name": S3_LOGS_FILE_NAME, "logs": latest_logs}

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

                post_logs_resp_json = post_logs_resp.json()

                if post_logs_resp.status_code != 200:
                    post_logs_error = (
                        post_logs_resp_json.get("detail")
                        if post_logs_resp_json.get("detail")
                        else ""
                    )
                    self.logger.error(
                        f"({post_logs_resp.status_code}) Failed to send cluster logs to Den: {post_logs_error}"
                    )
                else:
                    self.logger.info(
                        f"Successfully sent cluster logs to Den. Next status check will be in {round(interval_size / 60, 2)} minutes."
                    )
            except Exception as e:
                self.logger.error(
                    f"Sending cluster logs to den has failed: {e}. Please check cluster logs for more info."
                )
                self.logger.warning(
                    f"Temporarily increasing the interval between two consecutive log retrievals."
                    f"Next log retrieval will be in {round(INCREASED_INTERVAL / 60, 2)} minutes. "
                    f"For changing the interval size, please run cluster.restart_server(). "
                    f"Interval size will be set to {interval_size}"
                )
            finally:
                await asyncio.sleep(interval_size)

    def send_cluster_logs_to_den(self):
        asyncio.run(self.asend_cluster_logs_to_den())
