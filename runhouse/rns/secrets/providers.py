from enum import Enum

from runhouse.rns.secrets.aws_secrets import AWSSecrets
from runhouse.rns.secrets.azure_secrets import AzureSecrets
from runhouse.rns.secrets.gcp_secrets import GCPSecrets
from runhouse.rns.secrets.github_secrets import GitHubSecrets
from runhouse.rns.secrets.huggingface_secrets import HuggingFaceSecrets
from runhouse.rns.secrets.lambda_secrets import LambdaSecrets
from runhouse.rns.secrets.sky_secrets import SkySecrets
from runhouse.rns.secrets.ssh_secrets import SSHSecrets


class Providers(Enum):
    AWS = AWSSecrets()
    AZURE = AzureSecrets()
    GCP = GCPSecrets()
    HUGGINGFACE = HuggingFaceSecrets()
    LAMBDA = LambdaSecrets()
    SKY = SkySecrets()
    SSH = SSHSecrets()
    GITHUB = GitHubSecrets()
