from runhouse.resources.secrets.provider_secrets.api_key_secret import ApiKeySecret


class CohereSecret(ApiKeySecret):
    """
    .. note::
            To create an CohereSecret, please use the factory method :func:`provider_secret`
            with ``provider="cohere"``.
    """

    _PROVIDER = "cohere"
    _DEFAULT_ENV_VARS = {"api_key": "COHERE_API_KEY"}

    @staticmethod
    def from_config(config: dict, dryrun: bool = False):
        return CohereSecret(**config, dryrun=dryrun)
