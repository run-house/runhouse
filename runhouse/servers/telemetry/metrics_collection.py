import psutil


def get_gpu_usage():
    import GPUtil

    gpus = GPUtil.getGPUs()
    total_gpu_memory = sum(gpu.memoryTotal for gpu in gpus) if gpus else 0
    total_used_memory = sum(gpu.memoryUsed for gpu in gpus) if gpus else 0
    free_memory = total_gpu_memory - total_used_memory if total_gpu_memory else 0
    gpu_utilization_percent = (
        sum(gpu.load * 100 for gpu in gpus) / len(gpus) if gpus else 0
    )
    return {
        "total_memory": total_gpu_memory,
        "used_memory": total_used_memory,
        "free_memory": free_memory,
        "gpu_count": len(gpus),
        "utilization_percent": round(gpu_utilization_percent, 2),
    }


def get_cpu_usage():
    cpu_percent = psutil.cpu_percent()

    # Convert to MB
    total_cpu_memory = psutil.virtual_memory().total / (1024**2)
    used_cpu_memory = psutil.virtual_memory().used / (1024**2)
    free_cpu_memory = psutil.virtual_memory().free / (1024**2)

    return {
        "utilization_percent": cpu_percent,
        "total_memory": total_cpu_memory,
        "used_memory": used_cpu_memory,
        "free_memory": free_cpu_memory,
    }


def update_cpu_utilization(
    cpu_utilization_counter, cpu_memory_usage_gauge, cpu_free_memory_gauge
):
    # Get CPU utilization and memory usage data
    cpu_usage = get_cpu_usage()
    cpu_utilization_counter.add(
        cpu_usage["utilization_percent"], {"unit": "percentage"}
    )
    cpu_memory_usage_gauge.record(cpu_usage["used_memory"], {"unit": "MB"})
    cpu_free_memory_gauge.record(cpu_usage["free_memory"], {"unit": "MB"})


def update_gpu_utilization(
    gpu_utilization_counter, gpu_memory_usage_gauge, gpu_count_gauge
):
    # Function to add GPU utilization to the counter
    gpu_usage = get_gpu_usage()
    gpu_utilization_counter.add(
        gpu_usage["utilization_percent"], {"unit": "percentage"}
    )
    gpu_memory_usage_gauge.record(gpu_usage["used_memory"], {"unit": "MB"})
    gpu_count_gauge.record(gpu_usage["gpu_count"], {"unit": "count"})
