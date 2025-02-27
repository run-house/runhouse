import asyncio
import copy
import datetime
import json
import os
import threading
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import httpx

import requests

import runhouse

from runhouse.constants import (
    DEFAULT_AUTOSTOP_CHECK_INTERVAL,
    DEFAULT_LOG_LEVEL,
    DEFAULT_STATUS_CHECK_INTERVAL,
    GPU_COLLECTION_INTERVAL,
    INCREASED_GPU_COLLECTION_INTERVAL,
    INCREASED_STATUS_CHECK_INTERVAL,
    MAX_GPU_INFO_LEN,
    REDUCED_GPU_INFO_LEN,
    SERVER_LOGFILE,
    SERVER_LOGS_FILE_NAME,
)

from runhouse.globals import configs, obj_store, rns_client
from runhouse.logger import get_logger
from runhouse.resources.hardware import load_cluster_config_from_file
from runhouse.resources.hardware.ray_utils import kill_actors
from runhouse.resources.hardware.utils import is_gpu_cluster
from runhouse.rns.rns_client import ResourceStatusData
from runhouse.rns.utils.api import ResourceAccess
from runhouse.servers.autostop_helper import AutostopHelper
from runhouse.servers.http.auth import AuthCache
from runhouse.servers.http.http_utils import CreateProcessParams
from runhouse.utils import (
    ColoredFormatter,
    get_pid,
    parse_gpu_usage,
    ServletType,
    sync_function,
)

logger = get_logger(__name__)


class ClusterServletError(Exception):
    pass


class ClusterServlet:
    async def __init__(
        self, cluster_config: Optional[Dict[str, Any]] = None, *args, **kwargs
    ):
        import psutil

        # We do this here instead of at the start of the HTTP Server startup
        # because someone can be running `HTTPServer()` standalone in a test
        # and still want an initialized cluster config in the servlet.
        if not cluster_config:
            cluster_config = load_cluster_config_from_file()
        self._cluster_config: Optional[Dict[str, Any]] = cluster_config

        self._initialized_servlet_args: Dict[str, CreateProcessParams] = {}
        self._key_to_servlet_name: Dict[Any, str] = {}
        self._auth_cache: AuthCache = AuthCache()
        self._paths_to_prepend_in_new_processes = []
        self._env_vars_to_set_in_new_processes = {}
        self._node_servlet_names: List[str] = []
        self._cluster_name = self._cluster_config.get("name", None)
        self._cluster_uri = (
            rns_client.format_rns_address(self._cluster_name)
            if self._cluster_name
            else None
        )

        # will be used in the periodic loop which sends cluster logs to den. Since we are saving logs to the server.log
        # file asynchronously, there might be a use case where we want to send logs to den but the server.log has not
        # been created yet, since it is a freshly initialized cluster servlet. In this case, we will return an empty string.
        self._is_log_file_ready = False

        self._are_multinode_servlets_ready = False

        # will need this to make sure that we are deleting the correct cluster servlet when we call kill_actors
        self.name = kwargs.get("name", "cluster_servlet")

        self.autostop_helper = None
        if self._cluster_config.get("resource_subtype", None) == "OnDemandCluster":
            self.autostop_helper = AutostopHelper()

        self.pid = get_pid()
        self.process = psutil.Process(pid=self.pid)
        self.gpu_metrics = None  # will be updated only if this is a gpu cluster.
        # will be used when self.gpu_metrics will be updated by different threads.
        self.lock = threading.Lock()

        self._cluster_config["is_gpu"] = is_gpu_cluster()

        # creating periodic check loops
        if self._cluster_config.get("is_gpu"):
            logger.debug("Creating _periodic_gpu_check thread.")
            collect_gpu_thread = threading.Thread(
                target=self._periodic_gpu_check, daemon=True
            )
            collect_gpu_thread.start()

        logger.debug("Creating periodic_cluster_checks thread.")
        self.cluster_checks_thread = threading.Thread(
            target=self.periodic_cluster_checks, daemon=True
        )
        self.cluster_checks_thread.start()

        logger.debug("Creating autostop check thread.")
        self.autostop_check_thread = threading.Thread(
            target=self.periodic_autostop_check, daemon=True
        )
        self.autostop_check_thread.start()

    ##############################################
    # List of node servlet names
    ##############################################
    async def aget_node_servlet_names(self) -> List[str]:
        return self._node_servlet_names

    async def aset_node_servlet_names(self, node_servlet_names: List[str]):
        self._node_servlet_names = node_servlet_names
        self._are_multinode_servlets_ready = True

    ##############################################
    # Add to path and get path methods
    ##############################################
    async def aadd_path_to_prepend_in_new_processes(self, path: str):
        self._paths_to_prepend_in_new_processes.append(path)

    async def aget_paths_to_prepend_in_new_processes(self) -> List[str]:
        return self._paths_to_prepend_in_new_processes

    ##############################################
    # Env vars to set and get env vars methods
    ##############################################
    async def aset_env_vars_globally(self, env_vars: Dict[str, Any]):

        await asyncio.gather(
            *[
                obj_store.acall_servlet_method(
                    servlet_name,
                    "aset_env_vars",
                    env_vars,
                    use_servlet_cache=False,
                )
                for servlet_name in await self.aget_all_initialized_servlet_args()
            ]
        )

        await self.aadd_env_vars_to_set_in_new_processes(env_vars)

    async def aadd_env_vars_to_set_in_new_processes(self, env_vars: Dict[str, Any]):
        self._env_vars_to_set_in_new_processes.update(env_vars)

    async def aget_env_vars_to_set_in_new_processes(self) -> Dict[str, str]:
        return self._env_vars_to_set_in_new_processes

    ##############################################
    # Cluster config state storage methods
    ##############################################
    async def aget_cluster_config(self) -> Dict[str, Any]:
        return self._cluster_config

    async def aset_cluster_config(self, cluster_config: Dict[str, Any]):
        if "is_gpu" not in cluster_config.keys():
            cluster_config["is_gpu"] = is_gpu_cluster()

        self._cluster_config = cluster_config

        # Propagate the changes to all other process's obj_stores
        await asyncio.gather(
            *[
                obj_store.acall_servlet_method(
                    servlet_name,
                    "aset_cluster_config",
                    cluster_config,
                    use_servlet_cache=False,
                )
                for servlet_name in await self.aget_all_initialized_servlet_args()
            ]
        )

        return self._cluster_config

    async def aset_cluster_config_value(self, key: str, value: Any):
        if self.autostop_helper and key == "autostop_mins":
            await self.autostop_helper.set_autostop(value)
        self._cluster_config[key] = value

        # Propagate the changes to all other process's obj_stores
        await asyncio.gather(
            *[
                obj_store.acall_servlet_method(
                    servlet_name,
                    "aset_cluster_config_value",
                    key,
                    value,
                    use_servlet_cache=False,
                )
                for servlet_name in await self.aget_all_initialized_servlet_args()
            ]
        )

        return self._cluster_config

    ##############################################
    # Auth cache internal functions
    ##############################################
    async def aresource_access_level(
        self, token: str, resource_uri: str
    ) -> Union[str, None]:
        # If the token in this request matches that of the owner of the cluster,
        # they have access to everything
        config_token = configs.token
        if config_token:
            if config_token == token:
                return ResourceAccess.WRITE

            if resource_uri and rns_client.validate_cluster_token(
                cluster_token=token, cluster_uri=resource_uri
            ):
                return ResourceAccess.WRITE

        return self._auth_cache.lookup_access_level(token, resource_uri)

    async def aget_username(self, token: str) -> str:
        return self._auth_cache.get_username(token)

    async def ahas_resource_access(self, token: str, resource_uri=None) -> bool:
        """Checks whether user has read or write access to a given module saved on the cluster."""
        from runhouse.rns.utils.api import ResourceAccess

        if token is None or self._cluster_name is None:
            # If no token or cluster uri are provided assume no access
            return False

        cluster_access = await self.aresource_access_level(token, self._cluster_name)
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
    async def aadd_servlet_initialized_args(
        self, servlet_name: str, init_args: CreateProcessParams
    ):
        self._initialized_servlet_args[servlet_name] = init_args

    async def ais_servlet_name_initialized(self, servlet_name: str) -> bool:
        return servlet_name in self._initialized_servlet_args

    async def aget_all_initialized_servlet_args(self) -> Set[str]:
        return self._initialized_servlet_args

    async def aget_key_to_servlet_name_dict_keys(self) -> List[Any]:
        return list(self._key_to_servlet_name.keys())

    async def aget_key_to_servlet_name_dict(self) -> Dict[Any, str]:
        return self._key_to_servlet_name

    async def aget_servlet_name_for_key(self, key: Any) -> str:
        if self.autostop_helper:
            await self.autostop_helper.set_last_active_time_to_now()
        return self._key_to_servlet_name.get(key, None)

    async def aput_servlet_name_for_key(self, key: Any, servlet_name: str):
        if not await self.ais_servlet_name_initialized(servlet_name):
            raise ValueError(
                f"Env servlet name {servlet_name} not initialized, and you tried to mark a resource as in it."
            )
        self._key_to_servlet_name[key] = servlet_name

    async def apop_servlet_name_for_key(self, key: Any, *args) -> str:
        # *args allows us to pass default or not
        return self._key_to_servlet_name.pop(key, *args)

    async def aclear_key_to_servlet_name_dict(self):
        self._key_to_servlet_name = {}

    ##############################################
    # Remove Env Servlet
    ##############################################
    async def aclear_all_references_to_servlet_name(self, servlet_name: str):
        # using lock to prevent status thread access self._initialized_servlet_args before the env is deleted.
        with self.lock:
            del self._initialized_servlet_args[servlet_name]
            deleted_keys = [
                key
                for key, env in self._key_to_servlet_name.items()
                if env == servlet_name
            ]
            for key in deleted_keys:
                self._key_to_servlet_name.pop(key)

            # remove from servlet cache in all processes
            await asyncio.gather(
                *[
                    obj_store.acall_servlet_method(
                        servlet,
                        "aremove_servlet_from_cache",
                        servlet_name,
                    )
                    for servlet in await self.aget_all_initialized_servlet_args()
                ]
            )
            # current process
            await obj_store.adelete_servlet_from_cache(servlet_name)

        return deleted_keys

    ##############################################
    # Periodic Cluster Checks APIs
    ##############################################
    async def asave_status_metrics_to_den(self, status: dict):
        from runhouse.resources.hardware.utils import RunhouseDaemonStatus

        # making a copy so the status won't be modified with pop, since it will be returned after sending to den.
        # (status is passed as pointer).
        status_copy = copy.deepcopy(status)
        servlet_processes = status_copy.pop("processes")

        status_data = {
            "daemon_status": RunhouseDaemonStatus.RUNNING,
            "resource_type": status_copy.get("cluster_config").pop(
                "resource_type", "cluster"
            ),
            "resource_info": status_copy,
            "processes": servlet_processes,
        }

        client = httpx.AsyncClient()

        return await client.post(
            f"{rns_client.api_server_url}/resource/{self._cluster_uri}/cluster/status",
            data=json.dumps(status_data),
            headers=rns_client.request_headers(),
        )

    def save_status_metrics_to_den(self, status: dict):
        return sync_function(self.asave_status_metrics_to_den)(status)

    async def acheck_cluster_status(self, send_to_den: bool = True):
        logger.debug("Performing cluster status checks")
        status, den_resp_status_code = await self.astatus(send_to_den=send_to_den)

        if not send_to_den:
            return status, den_resp_status_code

        if den_resp_status_code == 404:
            logger.info(
                "Cluster has not yet been saved to Den, cannot update status or logs"
            )
        elif den_resp_status_code != 200:
            logger.error("Failed to send cluster status to Den")
        else:
            logger.debug("Successfully sent cluster status to Den")

        return status, den_resp_status_code

    async def acheck_cluster_logs(
        self, interval_size: int = DEFAULT_STATUS_CHECK_INTERVAL
    ):
        logger.debug("Performing logs checks")

        cluster_config = await self.aget_cluster_config()
        prev_end_log_line = cluster_config.get("end_log_line", 0)
        (
            logs_den_resp,
            new_start_log_line,
            new_end_log_line,
        ) = await self.send_cluster_logs_to_den(
            prev_end_log_line=prev_end_log_line,
        )
        if not logs_den_resp:
            logger.debug(
                f"No logs were generated in the past {interval_size} minute(s), logs were not sent to Den"
            )

        elif logs_den_resp.status_code == 200:
            logger.debug("Successfully sent cluster logs to Den")
            await self.aset_cluster_config_value(
                key="start_log_line", value=new_start_log_line
            )
            await self.aset_cluster_config_value(
                key="end_log_line", value=new_end_log_line
            )
        else:
            logger.error("Failed to send logs to Den")

        return logs_den_resp, new_start_log_line, new_end_log_line

    async def aperiodic_autostop_check(self):
        """Periodically check the autostop of the cluster"""
        autostop_interval = int(
            os.getenv("RH_AUTOSTOP_INTERVAL", DEFAULT_AUTOSTOP_CHECK_INTERVAL)
        )
        while True:
            try:
                should_update_autostop: bool = self.autostop_helper is not None
                if should_update_autostop:
                    status, _ = await self.acheck_cluster_status(send_to_den=False)
                    await self._update_autostop(status)
                    logger.debug("Successfully updated autostop")

            except Exception as e:
                if (
                    "Failed to look up actor with name" in str(e)
                    and not self._are_multinode_servlets_ready
                ):
                    # try to update autostop from a multinode cluster before the NodeServlets on all nodes is ready
                    continue
                logger.error(f"Autostop check has failed: {e}")

                # killing the cluster servlet only if log level is debug
                log_level = os.getenv("RH_LOG_LEVEL") or DEFAULT_LOG_LEVEL
                if log_level.lower() == "debug":
                    kill_actors(gracefully=False, actor_name=self.name)
                    break

            finally:
                logger.debug(f"Autostop interval set to {autostop_interval} seconds")
                await asyncio.sleep(autostop_interval)

    async def aperiodic_cluster_checks(self):
        """Periodically check the status of the cluster, gather metrics about the cluster's utilization & memory,
        and save it to Den."""

        logger.debug("started periodic cluster checks")
        disable_observability = (
            os.getenv("disable_observability", "false").strip().lower() == "true"
        )

        while True:
            should_send_status_and_logs_to_den: bool = (
                configs.token is not None and self._cluster_uri is not None
            )

            if not should_send_status_and_logs_to_den:
                break

            cluster_config = await self.aget_cluster_config()
            interval_size = cluster_config.get(
                "status_check_interval", DEFAULT_STATUS_CHECK_INTERVAL
            )

            try:
                status, den_resp_code = await self.acheck_cluster_status(
                    send_to_den=should_send_status_and_logs_to_den
                )

                if interval_size == -1 or not should_send_status_and_logs_to_den:
                    continue

                logger.debug("Performing cluster checks")

                if disable_observability:
                    logger.info(
                        "Cluster observability not enabled, skipping metrics collection."
                    )
                    break

                if den_resp_code == 404:
                    logger.info(
                        "Cluster has not yet been saved to Den, cannot update status or logs."
                    )
                elif den_resp_code != 200:
                    logger.error(
                        f"Failed to send cluster status to Den, status_code: {den_resp_code}"
                    )
                else:
                    logger.debug("Successfully sent cluster status to Den")

                    (
                        logs_resp_status_code,
                        new_start_log_line,
                        new_end_log_line,
                    ) = await self.acheck_cluster_logs(interval_size=interval_size)

                    if not logs_resp_status_code:
                        logger.debug(
                            f"No logs were generated in the past {interval_size} minute(s), logs were not sent to Den."
                        )

                    elif logs_resp_status_code == 200:
                        await self.aset_cluster_config_value(
                            key="start_log_line", value=new_start_log_line
                        )
                        await self.aset_cluster_config_value(
                            key="end_log_line", value=new_end_log_line
                        )

                        logger.debug("Successfully sent cluster logs to Den.")

            except Exception as e:
                if (
                    "Failed to look up actor with name" in str(e)
                    and not self._are_multinode_servlets_ready
                ):
                    # try to collect status from a multinode cluster before the NodeServlets on all nodes is ready
                    continue
                logger.error(f"Cluster checks have failed: {e}.\n")

                # killing the cluster servlet only if log level is debug
                log_level = os.getenv("RH_LOG_LEVEL") or DEFAULT_LOG_LEVEL
                if log_level.lower() == "debug":
                    kill_actors(gracefully=False, actor_name=self.name)

                else:
                    await asyncio.sleep(INCREASED_STATUS_CHECK_INTERVAL)

            finally:
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

    def periodic_autostop_check(self):
        asyncio.run(self.aperiodic_autostop_check())

    async def _update_autostop(self, status: dict):
        function_running = any(
            any(
                len(
                    resource["process_resource_mapping"][resource_name].get(
                        "active_function_calls", []
                    )
                )
                > 0
                for resource_name in resource["process_resource_mapping"].keys()
            )
            for resource in status.get("processes", {}).values()
        )
        if function_running:
            await self.autostop_helper.set_last_active_time_to_now()

        # We do this separately from the set_last_active_time_to_now call above because
        # function_running will only reflect activity from functions which happen to be running during
        # the status check. We still need to attempt to register activity for functions which have
        # been called and completed.
        await self.autostop_helper.register_activity_if_needed()

    async def _status_for_servlet(self, servlet_name):
        try:
            (
                objects_in_servlet,
                servlet_utilization_data,
            ) = await obj_store.acall_servlet_method(
                servlet_name, method="astatus_local"
            )

            return {
                "servlet_name": servlet_name,
                "objects_in_servlet": objects_in_servlet,
                "servlet_utilization_data": servlet_utilization_data,
            }

        # Need to catch the exception here because we're running this in a gather,
        # and need to know which env servlet failed
        except Exception as e:
            return {"servlet_name": servlet_name, "Exception": e}

    async def _aperiodic_gpu_check(self):
        """periodically collects cluster gpu usage"""
        import pynvml

        logger.debug("Started gpu usage collection")

        pynvml.nvmlInit()  # init nvidia ml info collection

        while True:
            try:
                gpu_count = pynvml.nvmlDeviceGetCount()
                with self.lock:
                    if not self.gpu_metrics:
                        self.gpu_metrics = {device: [] for device in range(gpu_count)}

                    for gpu_index in range(gpu_count):
                        handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
                        util_info = pynvml.nvmlDeviceGetUtilizationRates(handle)
                        memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)

                        total_memory = memory_info.total  # in bytes
                        used_memory = memory_info.used  # in bytes
                        free_memory = memory_info.free  # in bytes
                        utilization_percent = float(util_info.gpu)  # make it float

                        # to reduce cluster memory usage (we are saving the gpu_usage info on the cluster),
                        # we save only the most updated gpu usage. If for some reason the size of updated_gpu_info is
                        # too big, we remove the older gpu usage info.
                        # This is relevant when using cluster.status() directly and not relying on status being sent to den.
                        updated_gpu_info = self.gpu_metrics[gpu_index]
                        if len(updated_gpu_info) + 1 > MAX_GPU_INFO_LEN:
                            updated_gpu_info = updated_gpu_info[REDUCED_GPU_INFO_LEN:]
                        updated_gpu_info.append(
                            {
                                "total_memory": total_memory,
                                "used_memory": used_memory,
                                "free_memory": free_memory,
                                "utilization_percent": utilization_percent,
                            }
                        )
                        self.gpu_metrics[gpu_index] = updated_gpu_info

            except Exception as e:
                logger.error(
                    f"{self._cluster_name}'s GPU metrics collection failed: {str(e)}"
                )

                pynvml.nvmlShutdown()

                # killing the cluster servlet only if log level is debug
                log_level = os.getenv("RH_LOG_LEVEL") or DEFAULT_LOG_LEVEL
                if log_level.lower() == "debug":
                    kill_actors(gracefully=False, actor_name=self.name)
                else:
                    # increase sleep interval
                    await asyncio.sleep(INCREASED_GPU_COLLECTION_INTERVAL)

            finally:
                # collects gpu usage every 5 seconds.
                await asyncio.sleep(GPU_COLLECTION_INTERVAL)

    def _periodic_gpu_check(self):
        # This is only ever called once in its own thread, so we can do asyncio.run here instead of `sync_function`.
        asyncio.run(self._aperiodic_gpu_check())

    def _get_cluster_gpu_usage(
        self, is_multinode: bool, workers_usage: list, send_to_den: bool = False
    ):

        updated_workers_usage = []
        head_collected_gpus_info = copy.deepcopy(self.gpu_metrics)

        if not is_multinode and (
            not head_collected_gpus_info or not head_collected_gpus_info[0]
        ):
            return updated_workers_usage

        # Updating the head worker's resource usage to include its GPU usage.
        head_node_resource_usage = workers_usage[0]
        head_node_resource_usage["server_gpu_usage"] = parse_gpu_usage(
            collected_gpu_info=head_collected_gpus_info,
            servlet_type=ServletType.cluster,
        )
        # Adding the head node's resource usage as the first element in the workers' resources list.
        updated_workers_usage.append(head_node_resource_usage)

        # adding the gpu usage of the other node (if such exist) and make them to the first element of the worker list,
        # based on their order (equal to the ips list order)
        compute_properties = self._cluster_config.get("compute_properties", None)
        internal_ips = (
            compute_properties.get("internal_ips", None) if compute_properties else None
        )
        if is_multinode and internal_ips and self._are_multinode_servlets_ready:
            for ip_index in range(1, len(internal_ips)):
                worker_resource_usage = workers_usage[ip_index]
                if self._are_multinode_servlets_ready:
                    node_servlet_name = obj_store.node_servlet_name_for_ip(
                        internal_ips[ip_index]
                    )
                    node_servlet = obj_store.get_servlet(name=node_servlet_name)
                    worker_gpu_usage = obj_store.call_actor_method(
                        node_servlet, "get_gpu_metrics", send_to_den
                    )
                    worker_gpu_usage = parse_gpu_usage(
                        collected_gpu_info=worker_gpu_usage,
                        servlet_type=ServletType.cluster,
                    )
                    worker_resource_usage["server_gpu_usage"] = worker_gpu_usage

                updated_workers_usage.append(worker_resource_usage)

        return updated_workers_usage

    def _map_internal_external_ip(self, internal_ip: str):
        compute_properties = self._cluster_config.get("compute_properties", None)
        ips: list = (
            compute_properties.get("ips")
            if compute_properties
            else self._cluster_config.get("ips")
        )
        internal_ips: list = (
            compute_properties.get("internal_ips") if compute_properties else ips
        )

        return ips[internal_ips.index(internal_ip)]

    async def _aget_node_cpu_usage(self, internal_ip: str):
        return {
            "ip": self._map_internal_external_ip(internal_ip),
            "server_cpu_usage": await obj_store.acall_servlet_method(
                servlet_name=obj_store.node_servlet_name_for_ip(internal_ip),
                method="aget_cpu_usage",
            ),
        }

    async def astatus(self, send_to_den: bool = False) -> Tuple[Dict, Optional[int]]:
        import psutil

        config_cluster = copy.deepcopy(self._cluster_config)
        compute_properties = config_cluster.get("compute_properties", None)
        ips = (
            compute_properties.get("ips")
            if compute_properties
            else config_cluster.get("ips")
        )
        internal_ips = (
            compute_properties.get("internal_ips") if compute_properties else ips
        )
        is_multinode = len(ips) > 1
        head_ip = ips[0]

        # Popping out creds because we don't want to show them in the status
        config_cluster.pop("creds", None)

        # Getting data from each env servlet about the objects it contains and the utilization data
        servlet_utilization_data = {}
        with self.lock:
            servlets_status = await asyncio.gather(
                *[
                    self._status_for_servlet(servlet_name)
                    for servlet_name in await self.aget_all_initialized_servlet_args()
                ],
            )

        # Store the data for the appropriate env servlet name
        for process_status in servlets_status:
            servlet_name = process_status.get("servlet_name")

            # Nothing to store if there was an exception
            if "Exception" in process_status.keys():
                e = process_status.get("Exception")
                logger.warning(
                    f"Exception {str(e)} in status for env servlet {servlet_name}"
                )
                servlet_utilization_data[servlet_name] = {}

            else:
                # Store what was in the env and the utilization data
                process_memory_info = process_status.get("servlet_utilization_data")
                process_memory_info["process_resource_mapping"] = process_status.get(
                    "objects_in_servlet"
                )
                servlet_utilization_data[servlet_name] = process_memory_info

        # A dictionary that match the keys of psutil.virtual_memory()._asdict() to match the keys we expect in Den.
        relevant_memory_info = {
            "available": "free_memory",
            "percent": "used_memory_percent",
            "total": "total_memory",
            "used": "used_memory",
        }

        # Fields: `total`, `available`, `percent`, `used`, `free`, `active`, `inactive`, `buffers`, `cached`, `shared`, `slab`
        # according to psutil docs, percent = (total - available) / total * 100
        # we now it is memory usage of the head node because it is calculated on the cluster_servlet which is
        # initialized on the head node
        head_cpu_usage = psutil.virtual_memory()._asdict()

        head_cpu_usage = {v: head_cpu_usage[k] for k, v in relevant_memory_info.items()}

        head_cpu_usage["utilization_percent"] = psutil.cpu_percent(
            interval=0
        )  # value is between 0 and 100

        head_node_resource_usage = {
            "ip": head_ip,
            "server_cpu_usage": head_cpu_usage,
        }

        nodes_resource_usage = [head_node_resource_usage]

        # if the cluster is multinode, calculate the cpu_utilization and memory usage of the nodes that are not the head node.
        if is_multinode and ips and self._are_multinode_servlets_ready:
            workers_resource_usage = await asyncio.gather(
                *[
                    self._aget_node_cpu_usage(internal_ip=internal_ip)
                    for internal_ip in internal_ips[1:]
                ]
            )
            nodes_resource_usage = nodes_resource_usage + workers_resource_usage

        # get cluster gpu usage (if it's a multinode cluster, we get gpu usage per node)
        workers_resource_usage = (
            self._get_cluster_gpu_usage(
                is_multinode=is_multinode,
                workers_usage=nodes_resource_usage,
                send_to_den=send_to_den,
            )
            if self._cluster_config.get("is_gpu")
            else nodes_resource_usage
        )

        # rest the gpu_info only after the status was sent to den. If we should not send status to den,
        # self.gpu_metrics will not be updated at all, therefore should not be reset.
        if send_to_den:
            with self.lock:
                self.gpu_metrics = None

        status_data = {
            "cluster_config": config_cluster,
            "runhouse_version": runhouse.__version__,
            "server_pid": self.pid,
            "workers": workers_resource_usage,
            "is_multinode": is_multinode,
            "processes": servlet_utilization_data,
        }

        # converting status_data to ResourceStatusData instance to verify we constructed the status data correctly
        status_data = ResourceStatusData(**status_data).model_dump()

        if send_to_den:
            logger.debug("Sending cluster status to Den")
            den_resp = self.save_status_metrics_to_den(status=status_data)
            return status_data, den_resp.status_code

        return status_data, None

    def status(self, send_to_den: bool = False):
        return sync_function(self.astatus)(send_to_den=send_to_den)

    ##############################################
    # Save cluster logs to Den
    ##############################################
    def _get_logs(self):
        from pathlib import Path

        if not Path(SERVER_LOGFILE).exists():
            if not self._is_log_file_ready:
                self._is_log_file_ready = True
                return ""
            raise ValueError(f"{SERVER_LOGFILE} is not found, can't send logs to den")

        with open(SERVER_LOGFILE) as log_file:
            log_lines = log_file.readlines()
        cleaned_log_lines = [ColoredFormatter.format_log(line) for line in log_lines]
        return " ".join(cleaned_log_lines)

    def _generate_logs_file_name(self):
        current_timestamp = datetime.datetime.utcnow().strftime("%Y_%m_%d_%H_%M_%S")
        return f"{current_timestamp}_{SERVER_LOGS_FILE_NAME}"

    async def send_cluster_logs_to_den(
        self, prev_end_log_line: int
    ) -> Tuple[Optional[requests.Response], Optional[int], Optional[int]]:
        """Load the most recent logs from the server's log file and send them to Den."""
        # setting to a list, so it will be easier to get the end line num  + the logs delta to send to den.
        latest_logs = self._get_logs().split("\n")

        # minus 1 because we start counting logs from 0.
        new_end_log_line = len(latest_logs) - 1

        if new_end_log_line < prev_end_log_line:
            # Likely a sign that the daemon was restarted, so we should start from the beginning
            prev_end_log_line = 0

        # len(latest_logs) may equal to 0 if we are in the process of setting up the cluster servlet.
        # In this case the server.log might not have been created yet, therefore self._get_logs() returns an empty string ("").
        logs_to_den = (
            "\n".join(latest_logs[prev_end_log_line:])
            if len(latest_logs) > 0
            else latest_logs
        )

        if len(logs_to_den) == 0:
            return None, None, None

        logs_data = {
            "file_name": self._generate_logs_file_name(),
            "logs": logs_to_den,
            "start_line": prev_end_log_line,
            "end_line": new_end_log_line,
        }

        post_logs_resp = requests.post(
            f"{rns_client.api_server_url}/resource/{self._cluster_uri}/logs",
            data=json.dumps(logs_data),
            headers=rns_client.request_headers(),
        )

        resp_status_code = post_logs_resp.status_code
        if resp_status_code != 200:
            logger.error(
                f"{resp_status_code}: Failed to send cluster logs to Den: {post_logs_resp.json()}"
            )

        return post_logs_resp, prev_end_log_line, new_end_log_line
