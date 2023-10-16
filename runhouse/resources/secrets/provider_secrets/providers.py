from runhouse.resources.secrets.provider_secrets.aws_secret import AWSSecret
from runhouse.resources.secrets.provider_secrets.azure_secret import AzureSecret
from runhouse.resources.secrets.provider_secrets.gcp_secret import GCPSecret
from runhouse.resources.secrets.provider_secrets.github_secret import GitHubSecret
from runhouse.resources.secrets.provider_secrets.huggingface_secret import (
    HuggingFaceSecret,
)
from runhouse.resources.secrets.provider_secrets.lambda_secret import LambdaSecret
from runhouse.resources.secrets.provider_secrets.provider_secret import ProviderSecret

_str_to_provider_class = {
    "aws": AWSSecret,
    "gcp": GCPSecret,
    "lambda": LambdaSecret,
    "github": GitHubSecret,
    "huggingface": HuggingFaceSecret,
    "azure": AzureSecret,
}


def _get_provider_class(provider_str):
    if provider_str not in _str_to_provider_class:
        return ProviderSecret
    return _str_to_provider_class[provider_str]
