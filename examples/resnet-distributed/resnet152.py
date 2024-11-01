import boto3
import torch
import torch.nn as nn
from torchvision import models


class ResNet152Model(nn.Module):
    def __init__(self, num_classes=1000, pretrained=False, s3_bucket=None, s3_key=None):
        super(ResNet152Model, self).__init__()

        # Initialize the ResNet-152 model
        self.model = models.resnet152(pretrained=False)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

        # Load weights from S3 if specified
        if pretrained and s3_bucket and s3_key:
            self.load_weights_from_s3(s3_bucket, s3_key)

    def load_weights_from_s3(self, s3_bucket, s3_key, weights_path):
        s3 = boto3.client("s3")
        # Download the weights to a local file
        s3.download_file(s3_bucket, s3_key, weights_path)

        # Load the weights
        self.model.load_state_dict(torch.load(weights_path, map_location="cpu"))
        print("Pretrained weights loaded from S3.")

    def forward(self, x):
        return self.model(x)
