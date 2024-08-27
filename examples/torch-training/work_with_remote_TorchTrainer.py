# ## Working with remote objects
# You can run this code in a Python script, a Jupyter notebook, or any other Python environment. It will work while training is happening or long after it ends until the cluster is downed.
import runhouse as rh

# Define a cluster type - here we launch an on-demand AWS cluster with 1 NVIDIA A10G GPU.
# You can use any cloud you want, or existing compute
cluster = rh.ondemand_cluster(
    name="a10g-cluster", instance_type="A10G:1", provider="aws"
).up_if_not()

# Get our remote TorchTrainer by name
model = cluster.get("torch_model", default=None, remote=True)

# Get the training status of the model
print(model.return_status())

# Make a prediction with the model, which we can do even when training is happening in a different thread.
from torchvision import datasets, transforms

transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
)

local_dataset = datasets.MNIST(
    "./data", train=False, download=True, transform=transform
)
example_data, example_target = local_dataset[0][0].unsqueeze(0), local_dataset[0][1]
prediction = model.predict(example_data)
print(f"Predicted: {prediction}, Actual: {example_target}")
