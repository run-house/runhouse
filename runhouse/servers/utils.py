import copy
from enum import Enum


####################################################################################################
# Status collection utils
####################################################################################################
class ServletType(str, Enum):
    env = "env"
    cluster = "cluster"


def get_gpu_usage(server_pid: int):
    import subprocess

    gpu_general_info = (
        subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.total,memory.used,memory.free,count,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            stdout=subprocess.PIPE,
        )
        .stdout.decode("utf-8")
        .strip()
        .split(", ")
    )
    total_gpu_memory = int(gpu_general_info[0]) * (1024**2)  # in bytes
    total_used_memory = int(gpu_general_info[1]) * (1024**2)  # in bytes
    free_memory = int(gpu_general_info[2]) * (1024**2)  # in bytes
    gpu_count = int(gpu_general_info[3])
    gpu_utilization_percent = int(gpu_general_info[4]) / 100

    return {
        "total_memory": total_gpu_memory,
        "used_memory": total_used_memory,
        "free_memory": free_memory,
        "gpu_count": gpu_count,
        "gpu_utilization_percent": gpu_utilization_percent,
        "server_pid": server_pid,  # will be useful for multi-node clusters.
    }


def get_memory_usage():
    import psutil

    # Fields: `total`, `available`, `percent`, `used`, `free`, `active`, `inactive`, `buffers`, `cached`, `shared`,
    # `slab`. According to psutil docs, percent = (total - available) / total * 100
    memory_usage = psutil.virtual_memory()

    memory_usage_dict = {
        "used_memory": memory_usage.used,
        "total_memory": memory_usage.total,
        "percent": memory_usage.percent,
        "free_memory": memory_usage.free,
    }
    return memory_usage_dict


def get_cpu_utilization_percent(interval: int = 1):
    import psutil

    return psutil.cpu_percent(interval=interval)


def get_multinode_cpu_usage(
    memory_usage: dict, cpu_utilization: float, envs_servlet_utilization_data: dict
):
    updated_memory_usage, updated_cpu_utilization = {}, {}
    for env_name, env_data in envs_servlet_utilization_data.items():
        env_cpu_usage = env_data.get("env_cpu_usage")
        node_name = env_data.get("node_name")
        if "head" in node_name:
            updated_memory_usage[node_name] = copy.deepcopy(memory_usage)
            updated_cpu_utilization[node_name] = copy.deepcopy(cpu_utilization)
            continue
        elif node_name not in updated_memory_usage.keys():
            updated_memory_usage[node_name] = env_cpu_usage
            updated_cpu_utilization[node_name] = [
                updated_cpu_utilization[node_name]["utilization_percent"]
            ]
        else:
            updated_memory_usage[node_name]["used_memory"] += env_cpu_usage.get(
                "used_memory"
            )
            updated_cpu_utilization[node_name] = updated_cpu_utilization[
                node_name
            ].append(env_cpu_usage.get("utilization_percent"))

    for worker in updated_cpu_utilization:
        utilization_percent_info = updated_cpu_utilization[worker]
        average_utilization_percent = sum(utilization_percent_info) / len(
            utilization_percent_info
        )
        updated_cpu_utilization[worker] = average_utilization_percent

    return updated_memory_usage, updated_cpu_utilization


def get_multinode_gpu_usage(
    server_gpu_usage: dict, gpu_utilization: float, envs_servlet_utilization_data: dict
):
    # TODO: check if cpu_utilization is indeed float
    updated_gpu_memory_usage, updated_gpu_utilization = {}, {}
    for env_data in envs_servlet_utilization_data:
        env_gpu_usage = env_data.get("env_gpu_usage")
        node_name = env_data.get("node_name")
        if "head" in node_name:
            updated_gpu_memory_usage[node_name] = copy.deepcopy(server_gpu_usage)
            updated_gpu_utilization[node_name] = copy.deepcopy(gpu_utilization)
            continue
        elif node_name not in updated_gpu_memory_usage.keys():
            updated_gpu_memory_usage[node_name] = env_gpu_usage
            updated_gpu_utilization[node_name] = [
                updated_gpu_utilization[node_name]["utilization_percent"]
            ]
        else:
            updated_gpu_memory_usage[node_name]["used_memory"] += env_gpu_usage.get(
                "used_memory"
            )
            updated_gpu_utilization[node_name] = updated_gpu_utilization[
                node_name
            ].append(env_gpu_usage.get("utilization_percent"))

    for worker in updated_gpu_utilization:
        utilization_percent_info = updated_gpu_utilization[worker]
        average_utilization_percent = sum(utilization_percent_info) / len(
            utilization_percent_info
        )
        updated_gpu_utilization[worker] = average_utilization_percent
    return updated_gpu_memory_usage, updated_gpu_utilization
