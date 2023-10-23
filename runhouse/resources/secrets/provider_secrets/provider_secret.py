from typing import Dict, Optional

from runhouse.resources.secrets.secret import Secret


class ProviderSecret(Secret):
    _DEFAULT_CREDENTIALS_PATH = None
    _PROVIDER = None
    _ENV_VARS = {}

    def __init__(
        self,
        name: Optional[str] = None,
        provider: Optional[str] = None,
        values: Dict = {},
        path: str = None,
        env_vars: Dict = {},
        dryrun: bool = False,
        **kwargs,
    ):
        """
        Provider Secret class.

        .. note::
            To create a ProviderSecret, please use the factory method :func:`provider_secret`.
        """
        name = name or provider or self._PROVIDER
        self.provider = provider or self._PROVIDER
        path = path or self._DEFAULT_CREDENTIALS_PATH
        env_vars = env_vars or self._ENV_VARS
        super().__init__(
            name=name, values=values, path=path, env_vars=env_vars, dryrun=dryrun
        )

    @property
    def config_for_rns(self):
        config = super().config_for_rns
        config.update({"provider": self.provider})
        return config

    @staticmethod
    def from_config(config: dict, dryrun: bool = False):
        """Create a ProviderSecret object from a config dictionary."""
        return ProviderSecret(**config, dryrun=dryrun)

    # TODO
    # def _from_provider():
    #   uses built in provider tools to get secret values
    #   ex/ aws uses boto3 session, lambda has lambda cloud client
