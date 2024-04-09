from runhouse.resources.secrets.provider_secrets.api_key_secret import ApiKeySecret


class PineconeSecret(ApiKeySecret):
    """
    .. note::
            To create an PineconeSecret, please use the factory method :func:`provider_secret`
            with ``provider="pinecone"``.
    """

    _PROVIDER = "pinecone"
    _DEFAULT_ENV_VARS = {"api_key": "PINECONE_API_KEY"}

    @staticmethod
    def from_config(config: dict, dryrun: bool = False, _resolve_children: bool = True):
        return PineconeSecret(**config, dryrun=dryrun)
