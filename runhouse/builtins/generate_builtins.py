from runhouse.rns.hardware.skycluster import SkyCluster

rh_cpu = SkyCluster(
    name="^rh-cpu", instance_type="m5.large", provider="aws", dryrun=False
)
rh_8_cpu = SkyCluster(
    name="^rh-8-cpu", instance_type="m5.2xlarge", provider="aws", dryrun=False
)
rh_32_cpu = SkyCluster(
    name="^rh-32-cpu", instance_type="m5.8xlarge", provider="aws", dryrun=False
)
rh_gpu = SkyCluster(name="^rh-gpu", instance_type="K80:1", dryrun=False)
rh_4_gpu = SkyCluster(name="^rh-4-gpu", instance_type="K80:4", dryrun=False)
rh_8_gpu = SkyCluster(name="^rh-8-gpu", instance_type="K80:8", dryrun=False)
rh_v100 = SkyCluster(name="^rh-v100", instance_type="V100:1", dryrun=False)
rh_4_v100 = SkyCluster(name="^rh-4-v100", instance_type="V100:4", dryrun=False)
rh_8_v100 = SkyCluster(name="^rh-8-v100", instance_type="V100:8", dryrun=False)

# TODO make a way to save a config with arguments absent to leave them to user default.
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
    if "cpu" not in cluster.name:
        cluster.provider = None
    cluster.save()
