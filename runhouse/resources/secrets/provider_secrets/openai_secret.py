from runhouse.resources.secrets.provider_secrets.api_key_secret import ApiKeySecret


class OpenAISecret(ApiKeySecret):
    """
    .. note::
            To create an OpenAISecret, please use the factory method :func:`provider_secret` with ``provider="openai"``.
    """

    _PROVIDER = "openai"
    _DEFAULT_ENV_VARS = {"api_key": "OPENAI_API_KEY"}

    @staticmethod
    def from_config(config: dict, dryrun: bool = False, _resolve_children: bool = True):
        return OpenAISecret(**config, dryrun=dryrun)
