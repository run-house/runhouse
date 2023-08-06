import os
import pickle
import shutil
import tempfile
import textwrap

import pandas as pd

import pytest

import runhouse as rh
from runhouse.rns.api_utils.utils import create_s3_bucket


# https://docs.pytest.org/en/6.2.x/fixture.html#conftest-py-sharing-fixtures-across-multiple-files


@pytest.fixture
def blob_data():
    return pickle.dumps(list(range(50)))


# ----------------- Folders -----------------


@pytest.fixture
def local_folder(tmp_path):
    local_folder = rh.folder(path=tmp_path / "tests_tmp")
    local_folder.put({f"sample_file_{i}.txt": f"file{i}".encode() for i in range(3)})
    return local_folder


@pytest.fixture
def cluster_folder(cpu_cluster, local_folder):
    return local_folder.to(system=cpu_cluster)


@pytest.fixture
def s3_folder(local_folder):
    s3_folder = local_folder.to(system="s3")
    yield s3_folder

    # Delete files from S3
    s3_folder.rm()


@pytest.fixture
def gcs_folder(local_folder):
    gcs_folder = local_folder.to(system="gs")
    yield gcs_folder

    # Delete files from GCS
    gcs_folder.rm()


# ----------------- Tables -----------------
@pytest.fixture
def huggingface_table():
    from datasets import load_dataset

    dataset = load_dataset("yelp_review_full", split="train[:1%]")
    return dataset


@pytest.fixture
def arrow_table():
    import pyarrow as pa

    df = pd.DataFrame(
        {
            "int": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "str": ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"],
        }
    )
    arrow_table = pa.Table.from_pandas(df)
    return arrow_table


@pytest.fixture
def cudf_table():
    import cudf

    gdf = cudf.DataFrame(
        {"id": [1, 2, 3, 4, 5, 6], "grade": ["a", "b", "b", "a", "a", "e"]}
    )
    return gdf


@pytest.fixture
def pandas_table():
    df = pd.DataFrame(
        {"id": [1, 2, 3, 4, 5, 6], "grade": ["a", "b", "b", "a", "a", "e"]}
    )
    return df


@pytest.fixture
def dask_table():
    import dask.dataframe as dd

    index = pd.date_range("2021-09-01", periods=2400, freq="1H")
    df = pd.DataFrame({"a": range(2400), "b": list("abcaddbe" * 300)}, index=index)
    ddf = dd.from_pandas(df, npartitions=10)
    return ddf


@pytest.fixture
def ray_table():
    import ray

    ds = ray.data.range(10000)
    return ds


# ------------------ SageMaker -----------------


@pytest.fixture
def sm_entry_point():
    return "pytorch_train.py"


@pytest.fixture
def pytorch_training_script():
    training_script = textwrap.dedent(
        """\
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.utils.data as data
    import torchvision.datasets as datasets
    import torchvision.transforms as transforms


    class MyModel(nn.Module):
        def __init__(self):
            super(MyModel, self).__init__()
            # Define your model architecture here
            self.fc1 = nn.Linear(784, 128)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(128, 10)
            self.softmax = nn.Softmax(dim=1)

        def forward(self, x):
            x = self.fc1(x)
            x = self.relu(x)
            x = self.fc2(x)
            x = self.softmax(x)
            return x


    # Define your dataset class
    class MyDataset(data.Dataset):
        def __init__(self, data, targets):
            self.data = data
            self.targets = targets

        def __getitem__(self, index):
            x = self.data[index]
            y = self.targets[index]
            return x, y

        def __len__(self):
            return len(self.data)

    if __name__ == "__main__":
        # Set random seed for reproducibility
        torch.manual_seed(42)

        # Define hyperparameters
        learning_rate = 0.001
        batch_size = 64
        num_epochs = 10

        # Load the MNIST dataset and apply transformations
        train_dataset = datasets.MNIST(
            root="./data", train=True, transform=transforms.ToTensor(), download=True
        )

        # Extract the training data and targets
        train_data = (
                train_dataset.data.view(-1, 28 * 28).float() / 255.0
        )  # Flatten and normalize the input
        train_targets = train_dataset.targets

        dataset = MyDataset(train_data, train_targets)
        dataloader = data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Initialize your model
        model = MyModel()

        # Define loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        # Training loop
        for epoch in range(num_epochs):
            running_loss = 0.0
            for batch_idx, (inputs, targets) in enumerate(dataloader):
                # Zero the gradients
                optimizer.zero_grad()

                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, targets)

                # Backward pass and optimization
                loss.backward()
                optimizer.step()
            """
    )
    return training_script


@pytest.fixture
def sm_source_dir(sm_entry_point, pytorch_training_script):
    """Create the source directory and entry point needed for creating a SageMaker estimator"""

    # create tmp directory and file
    temp_dir = tempfile.mkdtemp()
    file_path = os.path.join(temp_dir, sm_entry_point)
    with open(file_path, "w") as file:
        file.write(pytorch_training_script)

    yield temp_dir

    # delete tmp directory and file
    os.remove(file_path)
    shutil.rmtree(temp_dir)


# ----------------- Clusters -----------------


@pytest.fixture
def cluster(request):
    """Parametrize over multiple fixtures - useful for running the same test on multiple hardware types."""
    # Example: @pytest.mark.parametrize("cluster", ["v100_gpu_cluster", "k80_gpu_cluster"], indirect=True)"""
    return request.getfixturevalue(request.param)


@pytest.fixture(scope="session")
def cpu_cluster():
    c = rh.ondemand_cluster("^rh-cpu")
    c.up_if_not()
    c.install_packages(["pytest"])
    # Save to RNS - to be loaded in other tests (ex: Runs)
    c.save()
    return c


@pytest.fixture(scope="session")
def sm_cluster():
    c = rh.sagemaker_cluster(name="rh-sagemaker", role=os.getenv("AWS_ROLE_ARN"))
    c.save()

    yield c

    c.teardown_and_delete()
    assert not c.is_up()


@pytest.fixture(scope="session")
def sm_gpu_cluster():
    c = rh.sagemaker_cluster(
        name="rh-sagemaker-gpu",
        instance_type="ml.g5.2xlarge",
        role=os.getenv("AWS_ROLE_ARN"),
    ).save()

    yield c

    c.teardown_and_delete()
    assert not c.is_up()


@pytest.fixture(scope="session")
def byo_cpu():
    # Spin up a new basic m5.xlarge EC2 instance
    c = (
        rh.ondemand_cluster(
            instance_type="m5.xlarge",
            provider="aws",
            region="us-east-1",
            image_id="ami-0a313d6098716f372",
            name="test-byo-cluster",
        )
        .up_if_not()
        .save()
    )

    c = rh.cluster(
        name="different-cluster", ips=[c.address], ssh_creds=c.ssh_creds()
    ).save()

    c.install_packages(["pytest"])
    c.send_secrets(["ssh"])

    return c


@pytest.fixture(scope="session")
def v100_gpu_cluster():
    return rh.ondemand_cluster(
        name="rh-v100", instance_type="V100:1", provider="aws"
    ).up_if_not()


@pytest.fixture(scope="session")
def k80_gpu_cluster():
    return rh.ondemand_cluster(
        name="rh-k80", instance_type="K80:1", provider="aws"
    ).up_if_not()


@pytest.fixture(scope="session")
def a10g_gpu_cluster():
    return rh.ondemand_cluster(
        name="rh-a10x", instance_type="g5.2xlarge", provider="aws"
    ).up_if_not()


# ----------------- Envs -----------------


@pytest.fixture
def test_env():
    return rh.env(["pytest"])


# ----------------- Packages -----------------


@pytest.fixture
def local_package(local_folder):
    return rh.package(path=local_folder.path, install_method="local")


@pytest.fixture
def s3_package(s3_folder):
    return rh.package(
        path=s3_folder.path, system=s3_folder.system, install_method="local"
    )


# ----------------- Functions -----------------
def summer(a: int, b: int):
    return a + b


def save_and_load_artifacts():
    cpu = rh.ondemand_cluster("^rh-cpu").save()
    loaded_cluster = rh.load(name=cpu.name)
    return loaded_cluster.name


def slow_running_func(a, b):
    import time

    time.sleep(20)
    return a + b


@pytest.fixture(scope="session")
def summer_func(cpu_cluster):
    return rh.function(summer, name="summer_func").to(cpu_cluster, env=["pytest"])


@pytest.fixture(scope="session")
def func_with_artifacts(cpu_cluster):
    return rh.function(save_and_load_artifacts, name="artifacts_func").to(
        cpu_cluster, env=["pytest"]
    )


@pytest.fixture(scope="session")
def slow_func(cpu_cluster):
    return rh.function(slow_running_func, name="slow_func").to(
        cpu_cluster, env=["pytest"]
    )


# ----------------- S3 -----------------
@pytest.fixture(scope="session")
def runs_s3_bucket():
    runs_bucket = create_s3_bucket("runhouse-runs")
    return runs_bucket.name


@pytest.fixture(scope="session")
def blob_s3_bucket():
    blob_bucket = create_s3_bucket("runhouse-blob")
    return blob_bucket.name


@pytest.fixture(scope="session")
def table_s3_bucket():
    table_bucket = create_s3_bucket("runhouse-table")
    return table_bucket.name


# ----------------- Runs -----------------


@pytest.fixture(scope="session")
def submitted_run(summer_func):
    """Initializes a Run, which will run synchronously on the cluster. Returns the function's result."""
    run_name = "synchronous_run"
    res = summer_func(1, 2, run_name=run_name)
    assert res == 3
    return run_name


@pytest.fixture(scope="session")
def submitted_async_run(summer_func):
    """Execute function async on the cluster. If a run already exists, do not re-run. Returns a Run object."""
    run_name = "async_run"
    async_run = summer_func.run(run_name=run_name, a=1, b=2)

    assert isinstance(async_run, rh.Run)
    return run_name
