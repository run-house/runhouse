from runhouse.resources.secrets.provider_secrets.api_key_secret import ApiKeySecret


class WandBSecret(ApiKeySecret):
    """
    .. note::
            To create an WandBSecret, please use the factory method :func:`provider_secret` with ``provider="wandb"``.
    """

    _PROVIDER = "wandb"
    _DEFAULT_ENV_VARS = {"api_key": "WANDB_API_KEY"}

    @staticmethod
    def from_config(config: dict, dryrun: bool = False, _resolve_children: bool = True):
        return WandBSecret(**config, dryrun=dryrun)
