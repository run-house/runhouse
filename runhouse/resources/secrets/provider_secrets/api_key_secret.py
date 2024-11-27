from typing import Dict, Optional, Union

from runhouse.resources.hardware.cluster import Cluster
from runhouse.resources.secrets.provider_secrets.provider_secret import ProviderSecret


class ApiKeySecret(ProviderSecret):
    """Secret class for providers consisting of a single API key, generally stored as an environment variable.

    .. note::
            To create an ApiKeySecret, please use the factory method :func:`provider_secret`
            and passing in the corresponding provider.
    """

    def write(
        self,
        file: bool = False,
        env: bool = False,
        path: str = None,
        env_vars: Dict = None,
        overwrite: bool = False,
        write_config: bool = True,
    ):
        if not file or path:
            env = True
        super().write(
            file=file,
            env=env,
            path=path,
            env_vars=env_vars,
            overwrite=overwrite,
            write_config=write_config,
        )

    def to(
        self,
        system: Union[str, Cluster],
        path: str = None,
        process: Optional[str] = None,
        values: bool = True,
        name: Optional[str] = None,
    ):
        return super().to(
            system=system, path=path, process=process, values=values, name=name
        )
