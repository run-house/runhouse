from runhouse.resources.secrets.provider_secrets.huggingface_secret import (
    HuggingFaceSecret,
)
from runhouse.resources.secrets.provider_secrets.provider_secret import ProviderSecret
from runhouse.resources.secrets.provider_secrets.ssh_secret import SSHSecret


_str_to_provider_class = {
    "huggingface": HuggingFaceSecret,
    "ssh": SSHSecret,
}


def _get_provider_class(provider_str):
    if provider_str not in _str_to_provider_class:
        return ProviderSecret
    return _str_to_provider_class[provider_str]
