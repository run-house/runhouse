# # A Basic Torch Image Classification Example with the MNIST Dataset


# Deploy a basic Torch model and training class to a remote GPU for training.
#
# We use the very popular MNIST dataset, which includes a large number
# of handwritten digits, and create a neural network that accurately identifies
# what digit is in an image.
import io

import boto3
import kubetorch as kt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# We can import functions and classes from our repo, and Kubetorch
# fill find and sync to remote from the Git root (or when finding a pyproject.toml / setup.py)
from my_simple_model import SimpleNN
from my_transforms import get_transform

from torch.utils.data import DataLoader
from torchvision import datasets

# ## Write Regular Python
# We now define our trainer class, which is regular Python; Kubetorch is agnostic to what
# you run within your program. In this example, the trainer class has methods to load data,
# train the model for one epoch, test the model on the test data, and then finally to save
# the model to S3.
class SimpleTrainer:
    def __init__(self, from_checkpoint=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SimpleNN().to(self.device)
        if from_checkpoint:
            self.model.load_state_dict(
                torch.load(from_checkpoint, map_location=self.device)
            )

        self.train_loader = None
        self.test_loader = None
        self.epoch = 0
        self.transform = get_transform()

    def load_data(self, path, batch_size, download=True):
        def mnist(is_train):
            return datasets.MNIST(
                path, train=is_train, download=download, transform=self.transform
            )

        self.train_loader = DataLoader(mnist(True), batch_size=batch_size, shuffle=True)
        self.test_loader = DataLoader(mnist(False), batch_size=batch_size)

    def train_model(self, learning_rate=0.001):
        self.model.to(self.device)
        self.model.train()

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        running_loss = 0.0

        for data, target in self.train_loader:
            data, target = data.to(self.device), target.to(self.device)
            optimizer.zero_grad()
            output = self.model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"{self.epoch + 1}, loss: {running_loss / 100:.3f}")

    def test_model(self):
        self.model.eval()
        total_loss, correct = 0, 0

        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                total_loss += F.cross_entropy(output, target, reduction="sum").item()
                correct += (output.argmax(1) == target).sum().item()

        n = len(self.test_loader.dataset)
        print(
            f"Test loss: {total_loss/n:.4f}, Accuracy: {correct}/{n} ({100. * correct/n:.2f}%)"
        )

    def predict(self, data):
        self.model.eval()
        with torch.no_grad():
            output = self.model(x.to(self.device))

        return output.argmax(1).item()

    def save_model(self, bucket, s3_path):
        try:
            buf = io.BytesIO()
            torch.save(self.model.state_dict(), buf)
            boto3.client("s3").upload_fileobj(buf.seek(0) or buf, bucket, s3_path)
            print("Uploaded checkpoint")
        except Exception:
            print("Did not upload checkpoint, might not be authorized")


# ## Use Kubetorch to Dispatch to Kubernetes
# Now, we define the main function that will run locally when we run this script, and send our
# our class to a remote gpu to run. We are in main, but this code can be run from anywhere (e.g.
# your orchestrator, in CI, from a notebook, etc.)
if __name__ == "__main__":
    gpu = kt.Compute(
        gpus="1",  # We request 1 GPU here; optionally also memory, cpu, etc
        image=kt.Image(
            image_id="nvcr.io/nvidia/pytorch:23.10-py3"
        ),  # You can also specify your own image
        inactivity_ttl="1h",  # Autostops the compute after 1 hour of inactivity
    )

    # We define our module and run it on the remote compute. We take our normal Python class SimpleTrainer, and wrap it in kt.cls()
    # Then, we use `.to()` to send it to the remote gpu we just defined. This deploys our code to remote; the first
    # time it runs, it might take up to a few minutes to allocate compute (autoscale up), pull images, and run. But
    # then any further runs where I change my Trainer class will be instantaneous
    model = kt.cls(SimpleTrainer).to(gpu)

    # We set some settings for the model training, and then call the remote service as if it were local.
    batch_size = 64
    epochs = 5
    learning_rate = 0.01

    model.load_data("./data", batch_size)

    for epoch in range(epochs):
        model.train_model(learning_rate=learning_rate)

        model.test_model()

        model.save_model(
            bucket_name="my-simple-torch-model-example",
            s3_file_path=f"checkpoints/model_epoch_{epoch + 1}.pth",
        )

    # We can now call inference against the model without redeploying it as a service; for instance
    # prediction = model.predict(example_data)
