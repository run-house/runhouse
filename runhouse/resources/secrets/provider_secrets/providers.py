from runhouse.resources.secrets.provider_secrets.anthropic_secret import AnthropicSecret
from runhouse.resources.secrets.provider_secrets.aws_secret import AWSSecret
from runhouse.resources.secrets.provider_secrets.azure_secret import AzureSecret
from runhouse.resources.secrets.provider_secrets.cohere_secret import CohereSecret
from runhouse.resources.secrets.provider_secrets.gcp_secret import GCPSecret
from runhouse.resources.secrets.provider_secrets.github_secret import GitHubSecret
from runhouse.resources.secrets.provider_secrets.huggingface_secret import (
    HuggingFaceSecret,
)
from runhouse.resources.secrets.provider_secrets.lambda_secret import LambdaSecret
from runhouse.resources.secrets.provider_secrets.langchain_secret import LangChainSecret
from runhouse.resources.secrets.provider_secrets.openai_secret import OpenAISecret
from runhouse.resources.secrets.provider_secrets.pinecone_secret import PineconeSecret
from runhouse.resources.secrets.provider_secrets.provider_secret import ProviderSecret
from runhouse.resources.secrets.provider_secrets.sky_secret import SkySecret
from runhouse.resources.secrets.provider_secrets.ssh_secret import SSHSecret
from runhouse.resources.secrets.provider_secrets.wandb_secret import WandBSecret


_str_to_provider_class = {
    # File and/or Env secrets
    "aws": AWSSecret,
    "azure": AzureSecret,
    "gcp": GCPSecret,
    "github": GitHubSecret,
    "huggingface": HuggingFaceSecret,
    "lambda": LambdaSecret,
    # SSH secrets
    "ssh": SSHSecret,
    "sky": SkySecret,
    # API key secrets
    "anthropic": AnthropicSecret,
    "cohere": CohereSecret,
    "langchain": LangChainSecret,
    "openai": OpenAISecret,
    "pinecone": PineconeSecret,
    "wandb": WandBSecret,
}


def _get_provider_class(provider_str):
    if provider_str not in _str_to_provider_class:
        return ProviderSecret
    return _str_to_provider_class[provider_str]
