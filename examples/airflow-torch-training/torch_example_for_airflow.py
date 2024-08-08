import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class TorchExampleBasic(nn.Module):
    def __init__(self):
        super(TorchExampleBasic, self).__init__()
        self.fc1 = nn.Linear(28*28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #    
        self.to(self.device)
        
    
    def forward(self, x):
        x = x.view(-1, 28*28)  # Flatten the input
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    

def DownloadData(path = './data'):
    train_dataset = datasets.MNIST(path, train=True, download=True)
    test_dataset = datasets.MNIST(path, train=False, download = True)


class SimpleTrainer():
    def __init__(self):
        super(SimpleTrainer, self).__init__()
        self.model = TorchExampleBasic() 
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
        self.epoch = 0 
        self.train_loader = None
        self.test_loader = None
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])


    def load_train(self, path, batch_size):
        data = datasets.MNIST(path, train=True, download=False, transform=self.transform)
        self.train_loader = DataLoader(data, batch_size=batch_size, shuffle=True)

    def load_test(self, path, batch_size):
        data = datasets.MNIST(path, train=False, download=False, transform=self.transform)
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
            
            if batch_idx % 100 == 99:    # print every 100 mini-batches
                print(f'[{self.epoch + 1}, {batch_idx + 1:5d}] loss: {running_loss / 100:.3f}')
                running_loss = 0.0

        print('Finished Training')
    
    def test_model(self):
        
        self.model.eval()
        test_loss = 0
        correct = 0
        
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                test_loss += F.cross_entropy(output, target, reduction='sum').item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
        
        test_loss /= len(self.test_loader.dataset)
        accuracy = 100. * correct / len(self.test_loader.dataset)
        
        print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(self.test_loader.dataset)} ({accuracy:.2f}%)\n')


    def predict(self, data):
        self.model.eval()
        with torch.no_grad():
            data = data.to(self.device)
            output = self.model(data)
            pred = output.argmax(dim=1, keepdim=True)
        return pred.item()
    
    def save_model(self, bucket_name, s3_file_path):
        try: ## Avoid failing if you're just trying the example and don't have S3 setup
            import io 
            import boto3

            buffer = io.BytesIO()
            torch.save(self.model.state_dict(), buffer)
            buffer.seek(0)  # Rewind the buffer to the beginning

            s3 = boto3.client('s3')
            s3.upload_fileobj(buffer, bucket_name, s3_file_path)
            print('uploaded checkpoint')
        except: 
            print('did not upload checkpoint')
