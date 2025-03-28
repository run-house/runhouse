# # A Basic Torch Image Classification Example with the MNIST Dataset


# Deploy a basic Torch model and training class to a remote GPU for training.
#
# We use the very popular MNIST dataset, which includes a large number
# of handwritten digits, and create a neural network that accurately identifies
# what digit is in an image.
#
# ## Setting up a model class
import kubetorch as kt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image

from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# Let's define a function that downloads the data. You can imagine this as a generic function to access data.
def download_data(path="./data"):
    datasets.MNIST(path, train=True, download=True)
    datasets.MNIST(path, train=False, download=True)
    print("Done with data download")


def preprocess_data(path):
    transform = transforms.Compose(
        [
            transforms.Resize(
                (28, 28), interpolation=Image.BILINEAR
            ),  # Resize to 28x28 using bilinear interpolation
            transforms.ToTensor(),
            transforms.Normalize(
                (0.5,), (0.5,)
            ),  # Normalize with mean=0.5, std=0.5 for general purposes
        ]
    )

    train = datasets.MNIST(path, train=False, download=False, transform=transform)
    test = datasets.MNIST(path, train=False, download=False, transform=transform)
    print("Done with data preprocessing")
    print(f"Number of training samples: {len(train)}")
    print(f"Number of test samples: {len(test)}")


# Next, we define a model class. We define a very basic feedforward neural network with three fully connected layers.
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    def __init__(self, from_checkpoint=None):
        self.model = SimpleNN()
        if from_checkpoint:
            self.model.load_state_dict(torch.load(from_checkpoint))

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.epoch = 0

        self.train_loader = None
        self.test_loader = None

        self.transform = transforms.Compose([transforms.ToTensor()])

        self.accuracy = None
        self.test_loss = None

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

        self.epoch = self.epoch + 1
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
        self.test_loss = test_loss
        self.accuracy = 100.0 * correct / len(self.test_loader.dataset)

        print(
            f"\nTest set: Average loss: {self.test_loss:.4f}, Accuracy: {correct}/{len(self.test_loader.dataset)} ({self.accuracy:.2f}%)\n"
        )

    def predict(self, data):
        self.model.eval()
        with torch.no_grad():
            data = data.to(self.device)
            output = self.model(data)
            pred = output.argmax(dim=1, keepdim=True)
        return pred.item()

    def save_model(self, bucket_name, s3_file_path):
        try:  ## Avoid failing if you're just trying the example. Need to setup S3 access.
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

    def return_status(self):
        status = {
            "epochs_trained": self.epoch,
            "loss_test": self.test_loss,
            "accuracy_test": self.accuracy,
        }

        return status


# Now, we define the main function that will run locally when we run this script, and send our
# our class to a remote gpu. First, we define compute of the desired instance type and provider.
# Our `instance_type` here is defined as `A10G:1`, which is the accelerator type and count that we need. We could
# alternatively specify a specific AWS instance type, such as `p3.2xlarge` or `g4dn.xlarge`.
if __name__ == "__main__":
    # Define the compute - here we launch an on-demand cluster with 1 NVIDIA A10G GPU.
    # You can further specify the cloud, other compute constraints, or use existing compute
    gpu = kt.Compute(gpus="A10G:1", image=kt.images.pytorch())

    # We define our module and run it on the remote compute. We take our normal Python class SimpleTrainer, and wrap it in kt.cls()
    # We also take our function DownloadData and send it to the remote compute as well
    # Then, we use `.to()` to send it to the remote gpu we just defined.
    model = kt.cls(SimpleTrainer).to(gpu).init()

    model.compute.run_python(download_data)
    model.compute.run_python(preprocess_data, path="./data")

    # We set some settings for the model training
    batch_size = 64
    epochs = 5
    learning_rate = 0.01

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

    # We can now call inference against the model without redeploying it as a service
    # prediction = model.predict(example_data) # Assume you are defining example_data
