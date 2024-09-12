import runhouse as rh

# We check if we have already created a "rh_finetuner" on the remote which is an *instance* of the remote fine tuner class
cluster = rh.cluster(
    name="rh-a10x",
    instance_type="A10G:1",
    memory="32+",
    provider="aws",
).up_if_not()

fine_tuner_remote_name = "rh_finetuner"
fine_tuner_remote = cluster.get(fine_tuner_remote_name, default=None, remote=True)

# Check what the training status is on remote
if fine_tuner_remote is not None:
    print(fine_tuner_remote.get_training_status())
