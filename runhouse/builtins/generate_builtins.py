from runhouse.resources.hardware.on_demand_cluster import OnDemandCluster

rh_cpu = OnDemandCluster(name="^rh-cpu", instance_type="CPU:1", dryrun=False)
rh_8_cpu = OnDemandCluster(name="^rh-8-cpu", instance_type="CPU:8", dryrun=False)
rh_32_cpu = OnDemandCluster(name="^rh-32-cpu", instance_type="CPU:32", dryrun=False)
rh_gpu = OnDemandCluster(name="^rh-gpu", instance_type="K80:1", dryrun=False)
rh_4_gpu = OnDemandCluster(name="^rh-4-gpu", instance_type="K80:4", dryrun=False)
rh_8_gpu = OnDemandCluster(name="^rh-8-gpu", instance_type="K80:8", dryrun=False)
rh_v100 = OnDemandCluster(name="^rh-v100", instance_type="V100:1", dryrun=False)
rh_4_v100 = OnDemandCluster(name="^rh-4-v100", instance_type="V100:4", dryrun=False)
rh_8_v100 = OnDemandCluster(name="^rh-8-v100", instance_type="V100:8", dryrun=False)

for cluster in [
    rh_cpu,
    rh_8_cpu,
    rh_32_cpu,
    rh_gpu,
    rh_4_gpu,
    rh_8_gpu,
    rh_v100,
    rh_4_v100,
    rh_8_v100,
]:
    cluster.autostop_mins = None
    cluster.provider = None
    # Need to manually more into builtins because we can't save there
    cluster.save(name=f"~/{cluster.name}")
