import textwrap

import pytest

import runhouse as rh

from ....conftest import init_args
from ....utils import test_env


######## Constants ########


######## Fixtures #########


@pytest.fixture(scope="session")
def sagemaker_cluster(request):
    return request.getfixturevalue(request.param)


@pytest.fixture(scope="session")
def sm_cluster():
    c = (
        rh.sagemaker_cluster(
            name="rh-sagemaker",
            profile="sagemaker",
        )
        .up_if_not()
        .save()
    )

    test_env().to(c)

    return c


@pytest.fixture(scope="session")
def sm_cluster_with_auth():
    args = {"name": "^rh-sagemaker-den-auth", "profile": "sagemaker", "den_auth": True}
    c = rh.sagemaker_cluster(**args).up_if_not().save()
    init_args[id(c)] = args

    test_env().to(c)

    return c


@pytest.fixture(scope="session")
def other_sm_cluster():
    c = (
        rh.sagemaker_cluster(name="rh-sagemaker-2", profile="sagemaker")
        .up_if_not()
        .save()
    )
    test_env().to(c)
    return c


@pytest.fixture(scope="session")
def sm_gpu_cluster():
    c = (
        rh.sagemaker_cluster(
            name="rh-sagemaker-gpu", instance_type="ml.g5.2xlarge", profile="sagemaker"
        )
        .up_if_not()
        .save()
    )
    test_env().to(c)

    return c


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
def sm_source_dir(sm_entry_point, pytorch_training_script, tmp_path):
    """Create the source directory and entry point needed for creating a SageMaker estimator"""

    # create tmp directory and file
    file_path = tmp_path / sm_entry_point

    with open(file_path, "w") as f:
        f.write(pytorch_training_script)

    return file_path.parent


@pytest.fixture(scope="session")
def summer_func_sm_auth(sm_cluster_with_auth):
    from tests.test_resources.test_modules.test_functions.conftest import summer

    return rh.function(summer, name="summer_func").to(
        sm_cluster_with_auth, env=["pytest"]
    )
