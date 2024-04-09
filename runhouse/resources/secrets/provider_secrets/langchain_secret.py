from runhouse.resources.secrets.provider_secrets.api_key_secret import ApiKeySecret


class LangChainSecret(ApiKeySecret):
    """
    .. note::
            To create an LangChainSecret, please use the factory method :func:`provider_secret`
            with ``provider="langchain"``.
    """

    _PROVIDER = "langchain"
    _DEFAULT_ENV_VARS = {"api_key": "LANGCHAIN_API_KEY"}

    @staticmethod
    def from_config(config: dict, dryrun: bool = False, _resolve_children: bool = True):
        return LangChainSecret(**config, dryrun=dryrun)
