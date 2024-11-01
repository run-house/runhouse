import torch
from resnet152 import ResNet152Model
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR


class ResNet152Trainer:
    def __init__(self, s3_bucket, s3_path):
        self.rank = None
        self.device_id = None
        self.device = None
        self.model = None

        self.criterion = None
        self.optimizer = None
        self.scheduler = None

        self.train_loader = None
        self.val_loader = None

        self.s3_bucket = s3_bucket
        self.s3_path = s3_path

        print("Remote class initialized")

    def init_comms(self):
        torch.distributed.init_process_group(backend="nccl")

        if torch.distributed.is_initialized():
            self.rank = torch.distributed.get_rank()
        else:
            self.rank = 0

        self.device_id = self.rank % torch.cuda.device_count()
        self.device = torch.device(f"cuda:{self.device_id}")

    def init_model(
        self, num_classes, model_weight_path, lr, weight_decay, step_size, gamma
    ):
        if model_weight_path:
            self.model = DDP(
                ResNet152Model(
                    num_classes=num_classes,
                    pretrained=True,
                    s3_bucket=self.s3_bucket,
                    s3_key=model_weight_path,
                ).to(self.device),
                device_ids=[self.device_id],
            )
        else:
            self.model = DDP(
                ResNet152Model(num_classes=num_classes).to(self.device),
                device_ids=[self.device_id],
            )

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = StepLR(self.optimizer, step_size=step_size, gamma=gamma)

    def load_train(self, path, cache_dir=None):
        import os

        from datasets import load_from_disk
        from torch.utils.data import DataLoader
        from torch.utils.data.distributed import DistributedSampler

        if cache_dir:
            if not os.path.exists(os.path.expanduser(cache_dir)):
                dataset = load_from_disk(path)
                dataset.save_to_disk(cache_dir)
            else:
                dataset = load_from_disk(cache_dir)
        else:
            dataset = load_from_disk(path)

        dataset = dataset.with_format("torch")
        sampler = DistributedSampler(dataset)
        self.train_loader = DataLoader(
            dataset, batch_size=32, shuffle=False, sampler=sampler
        )

    def load_test(self, path, cache_dir=None):
        import os

        from datasets import load_from_disk
        from torch.utils.data import DataLoader
        from torch.utils.data.distributed import DistributedSampler

        if cache_dir:
            if not os.path.exists(os.path.expanduser(cache_dir)):
                dataset = load_from_disk(path)
                dataset.save_to_disk(cache_dir)
            else:
                dataset = load_from_disk(cache_dir)
        else:
            dataset = load_from_disk(path)

        dataset = dataset.with_format("torch")
        sampler = DistributedSampler(dataset)
        self.val_loader = DataLoader(
            dataset, batch_size=32, shuffle=False, sampler=sampler
        )

    def train_epoch(self):
        self.model.train()
        running_loss = 0.0
        num_batches = len(self.train_loader)
        print_interval = max(
            1, num_batches // 10
        )  # Adjust this as needed, here set to every 10% of batches

        for batch_idx, batch in enumerate(self.train_loader):
            images = batch["image"].to(self.device)
            labels = batch["label"].to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()

            # Print progress every `print_interval` batches
            if (batch_idx + 1) % print_interval == 0:
                print(f"Batch {batch_idx + 1}/{num_batches}, Loss: {loss.item():.4f}")

        avg_loss = running_loss / num_batches
        return avg_loss

    def validate_epoch(self):
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.train_loader):
                images = batch["image"].to(self.device)
                labels = batch["label"].to(self.device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = correct / total
        return accuracy

    def train(
        self,
        num_classes,
        num_epochs,
        train_data_path,
        val_data_path,
        lr=1e-4,
        weight_decay=1e-4,
        step_size=7,
        gamma=0.1,
        model_weights_path=None,
    ):
        # Initialize distributed training
        self.init_comms()
        print("Remote comms initialized")
        self.init_model(
            num_classes, model_weights_path, lr, weight_decay, step_size, gamma
        )
        print("Model initialized")

        # Load training and validation data
        self.load_train(train_data_path)
        self.load_test(val_data_path)
        print("Data loaded")

        # Train the model
        for epoch in range(num_epochs):
            print(f"entering epoch {epoch}")
            train_loss = self.train_epoch()
            print(f"validating {epoch}")
            val_accuracy = self.validate_epoch()
            print(f"scheduler stepping {epoch}")
            self.scheduler.step()
            print(
                f"Epoch {epoch+1}/{num_epochs}, Loss: {train_loss:.4f}, Val Accuracy: {val_accuracy:.4f}"
            )
            # Save checkpoint every few epochs or based on validation performance
            if (epoch == 0 or (epoch + 1) % 5 == 0) and self.rank == 0:
                print("Saving checkpoint")
                self.save(f"resnet152_epoch_{epoch+1}.pth")

    def save(self, name):
        import boto3

        torch.save(self.model.state_dict(), name)
        s3 = boto3.client("s3")
        s3.upload_file(name, self.s3_bucket, self.s3_path + "checkpoints/" + name)
        print(f"Model saved to s3://{self.s3_bucket}/{self.s3_path}checkpoints/{name}")

    def cleanup(self):
        torch.distributed.destroy_process_group()

    # Return a prediction with the model, pass in a PIL image
    def predict(self, image):
        from torchvision import transforms

        preprocess = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        image = preprocess(image).unsqueeze(0).to(self.device)

        self.model.eval()
        with torch.no_grad():
            output = self.model(image)
            _, predicted = torch.max(output, 1)
            return predicted.item()
