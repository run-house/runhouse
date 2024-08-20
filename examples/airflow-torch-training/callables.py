import runhouse as rh
from runhouse.logger import get_logger

from torch_example_for_airflow import DownloadData, SimpleTrainer


logger = get_logger(name=__name__)


def bring_up_cluster_callable(**kwargs):
    logger.info("Connecting to remote cluster")
    rh.ondemand_cluster(
        name="a10g-cluster", instance_type="A10G:1", provider="aws"
    ).up_if_not()
    # cluster.save() ## Use if you have a Runhouse Den account to save and monitor the resource.


def access_data_callable(**kwargs):
    logger.info("Step 2: Access data")
    env = rh.env(name="test_env", reqs=["torch", "torchvision"])

    cluster = rh.cluster(name="a10g-cluster").up_if_not()
    remote_download = rh.function(DownloadData).to(cluster, env=env)
    logger.info("Download function sent to remote")
    remote_download()
    logger.info("Downloaded")


def train_model_callable(**kwargs):
    cluster = rh.cluster(name="a10g-cluster").up_if_not()

    env = rh.env(name="test_env", reqs=["torch", "torchvision"])

    remote_torch_example = rh.module(SimpleTrainer).to(
        cluster, env=env, name="torch-basic-training"
    )

    model = remote_torch_example()

    batch_size = 64
    epochs = 5
    learning_rate = 0.01

    model.load_train("./data", batch_size)
    model.load_test("./data", batch_size)

    for epoch in range(epochs):
        model.train_model(learning_rate=learning_rate)
        model.test_model()
        model.save_model(
            bucket_name="my-simple-torch-model-example",
            s3_file_path=f"checkpoints/model_epoch_{epoch + 1}.pth",
        )


def down_cluster(**kwargs):
    cluster = rh.cluster(name="a10g-cluster")
    cluster.teardown()
