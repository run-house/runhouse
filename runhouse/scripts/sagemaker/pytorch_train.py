# start an SSM agent that will connect the training instance to AWS Systems Manager.
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.transforms as transforms

# Set random seed for reproducibility
torch.manual_seed(42)

# Define hyperparameters
learning_rate = 0.001
batch_size = 64
num_epochs = 10


# Define your custom neural network model
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
