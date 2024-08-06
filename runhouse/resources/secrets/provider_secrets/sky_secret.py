from typing import Dict, Optional

from runhouse.resources.secrets.provider_secrets.ssh_secret import SSHSecret


class SkySecret(SSHSecret):
    """
    .. note::
            To create a SkySecret, please use the factory method :func:`provider_secret` with ``provider="sky"``.
    """

    _PROVIDER = "sky"
    _DEFAULT_KEY = "sky-key"

    def __init__(
        self,
        name: Optional[str] = None,
        provider: Optional[str] = None,
        values: Dict = {},
        path: str = None,
        dryrun: bool = True,
        **kwargs,
    ):
        super().__init__(
            name=name, provider=provider, values=values, path=path, dryrun=dryrun
        )

    @staticmethod
    def from_config(config: dict, dryrun: bool = False, _resolve_children: bool = True):
        return SkySecret(**config, dryrun=dryrun)
