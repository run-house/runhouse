# # A Basic Torch Image Classification Example with the MNIST Dataset


# This example demonstrates how to deploy a basic Torch model
# and training class to AWS EC2 using Runhouse, and do the training.
#
# We use the very popular MNIST dataset which includes a large number
# of handwritten digits, and create a neural network that accurately identifies
# what digit is in an image.
#
# ## Setup credentials and dependencies
#
# Optionally, set up a virtual environment:
# ```shell
# $ conda create -n demo-runhouse python=3.10
# $ conda activate demo-runhouse
# ```
# Install the required dependencies:
# ```shell
# $ pip install "runhouse[aws]" torch torchvision
# ```
#
# We'll be launching an AWS EC2 instance via [SkyPilot](https://github.com/skypilot-org/skypilot), so we need to
# make sure our AWS credentials are set up:
# ```shell
# $ aws configure
# $ sky check
# ```
#
# ## Setting up a model class
import runhouse as rh
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# Let's define a function that downloads the data. You can imagine this as a generic function to access data.
def DownloadData(path="./data"):
    datasets.MNIST(path, train=True, download=True)
    datasets.MNIST(path, train=False, download=True)
    print("Done with data download")


# Next, we define a model class. We define a very basic feedforward neural network with three fully connected layers.
class TorchExampleBasic(nn.Module):
    def __init__(self):
        super(TorchExampleBasic, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  #

    def forward(self, x):
        x = x.view(-1, 28 * 28)  # Flatten the input
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# We also define a trainer class. The trainer class has methods to load data,
# train the model for one epoch, test the model on the test data, and then finally to save
# the model to S3.
class SimpleTrainer:
    def __init__(self):
        super(SimpleTrainer, self).__init__()
        self.model = TorchExampleBasic()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.epoch = 0

        self.train_loader = None
        self.test_loader = None

        self.transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )

    def load_train(self, path, batch_size):
        data = datasets.MNIST(
            path, train=True, download=False, transform=self.transform
        )
        self.train_loader = DataLoader(data, batch_size=batch_size, shuffle=True)

    def load_test(self, path, batch_size):
        data = datasets.MNIST(
            path, train=False, download=False, transform=self.transform
        )
        self.test_loader = DataLoader(data, batch_size=batch_size, shuffle=True)

    def train_model(self, learning_rate=0.001):
        self.model.train()

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        running_loss = 0.0

        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            optimizer.zero_grad()
            output = self.model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if batch_idx % 100 == 99:  # print every 100 mini-batches
                print(
                    f"[{self.epoch + 1}, {batch_idx + 1:5d}] loss: {running_loss / 100:.3f}"
                )
                running_loss = 0.0

        print("Finished Training")

    def test_model(self):

        self.model.eval()
        test_loss = 0
        correct = 0

        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                test_loss += F.cross_entropy(output, target, reduction="sum").item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(self.test_loader.dataset)
        accuracy = 100.0 * correct / len(self.test_loader.dataset)

        print(
            f"\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(self.test_loader.dataset)} ({accuracy:.2f}%)\n"
        )

    def predict(self, data):
        self.model.eval()
        with torch.no_grad():
            data = data.to(self.device)
            output = self.model(data)
            pred = output.argmax(dim=1, keepdim=True)
        return pred.item()

    def save_model(self, bucket_name, s3_file_path):
        try:  ## Avoid failing if you're just trying the example and don't have S3 setup
            import io

            import boto3

            buffer = io.BytesIO()
            torch.save(self.model.state_dict(), buffer)
            buffer.seek(0)  # Rewind the buffer to the beginning

            s3 = boto3.client("s3")
            s3.upload_fileobj(buffer, bucket_name, s3_file_path)
            print("uploaded checkpoint")
        except:
            print("did not upload checkpoint")


# ## Setting up Runhouse to run the defined class and functions remotely. 
#
# Now, we define the main function that will run locally when we run this script, and set up
# our Runhouse module on a remote cluster. First, we create a cluster with the desired instance type and provider.
# Our `instance_type` here is defined as `A10G:1`, which is the accelerator type and count that we need. We could
# alternatively specify a specific AWS instance type, such as `p3.2xlarge` or `g4dn.xlarge`.
#
# Learn more in the [Runhouse docs on clusters](/docs/tutorials/api-clusters).
#
# :::note{.info title="Note"}
# Make sure that your code runs within a `if __name__ == "__main__":` block, as shown below. Otherwise,
# the script code will run when Runhouse attempts to run code remotely.
#
# :::

if __name__ == "__main__":

    # Define a cluster type - here we launch an on-demand AWS cluster with 1 NVIDIA A10G GPU.
    # You can use any cloud you want, or existing compute
    cluster = rh.ondemand_cluster(
        name="a10g-cluster", instance_type="A10G:1", provider="aws"
    ).up_if_not()

    # Next, we define the environment for our module. This includes the required dependencies that need
    # to be installed on the remote machine, as well as any secrets (not needed here) that need to be synced up from local to remote.
    env = rh.env(name="test_env", reqs=["torch", "torchvision"])

    # We define our module and run it on the remote cluster. We take our normal Python class SimpleTrainer, and wrap it in rh.module()
    # We also take our function DownloadData and send it to the remote cluster as well
    # Then, we use `.to()` to send it to the remote cluster we just defined.
    #
    # Note that we also pass the `env` object to the `get_or_to` method, which will ensure that the environment is
    # set up on the remote machine before the module is run.
    remote_torch_example = rh.module(SimpleTrainer).to(
        cluster, env=env, name="torch-basic-training"
    )
    remote_download = rh.function(DownloadData).to(cluster, env=env)

    # ## Calling our remote Trainer
    # We instantiate the remote class
    model = remote_torch_example()  # Instantiating it based on the remote RH module

    # Though we could just as easily run identical code on local if my machine is capable of handling it.
    # model = SimpleTrainer()       # If instantiating a local example

    # We set some settings for the model training
    batch_size = 64
    epochs = 5
    learning_rate = 0.01

    # We create the datasets remotely, and then send them to the remote model / remote .load_train() method. The "preprocessing" happens remotely.
    # They become instance variables of the remote Trainer.
    remote_download()

    model.load_train("./data", batch_size)
    model.load_test("./data", batch_size)

    # We can train the model per epoch, use the remote .test() method to assess the accuracy, and save it from remote to a S3 bucket.
    # All errors, prints, and logs are sent to me as if I were debugging on my local machine, but all the work is done in the cloud.
    for epoch in range(epochs):
        # Train one epoch and test the model
        model.train_model(learning_rate=learning_rate)
        model.test_model()

        # Save each model checkpoint to S3
        model.save_model(
            bucket_name="my-simple-torch-model-example",
            s3_file_path=f"checkpoints/model_epoch_{epoch + 1}.pth",
        )

    # Finally, let's just see one prediction, as an example of using the remote model for inference.
    # We have in essence done the research, and in one breath, debugged the production pipeline and deployed a microservice.

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    test_dataset = datasets.MNIST("./data", train=False, transform=transform)
    example_data, example_target = test_dataset[0][0].unsqueeze(0), test_dataset[0][1]
    prediction = model.predict(example_data)
    print(f"Predicted: {prediction}, Actual: {example_target}")
