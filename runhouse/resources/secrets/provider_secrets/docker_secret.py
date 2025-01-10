from runhouse.resources.secrets.provider_secrets.provider_secret import ProviderSecret


class DockerRegistrySecret(ProviderSecret):
    """
    .. note::
        To create a DockerRegistrySecret, please use the factory method
        :func:`provider_secret` with ``provider="docker"``.
    """

    _PROVIDER = "docker"
    _DEFAULT_ENV_VARS = {
        "username": "SKYPILOT_DOCKER_USERNAME",
        "password": "SKYPILOT_DOCKER_PASSWORD",
        "server": "SKYPILOT_DOCKER_SERVER",
    }

    @staticmethod
    def from_config(config: dict, dryrun: bool = False, _resolve_children: bool = True):
        return DockerRegistrySecret(**config, dryrun=dryrun)
