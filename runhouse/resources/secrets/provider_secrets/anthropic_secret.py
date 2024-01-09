from runhouse.resources.secrets.provider_secrets.api_key_secret import ApiKeySecret


class AnthropicSecret(ApiKeySecret):
    """
    .. note::
            To create an AnthropicSecret, please use the factory method :func:`provider_secret`
            with ``provider="anthropic"``.
    """

    _PROVIDER = "anthropic"
    _DEFAULT_ENV_VARS = {"api_key": "ANTHROPIC_API_KEY"}

    @staticmethod
    def from_config(config: dict, dryrun: bool = False):
        return AnthropicSecret(**config, dryrun=dryrun)
