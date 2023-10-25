from pathlib import Path
from typing import Dict, List, Optional, Union

from runhouse.resources.secrets.env_secret import EnvSecret
from runhouse.resources.secrets.provider_secrets.providers import _get_provider_class

from runhouse.resources.secrets.secret import Secret


def secret(
    name: Optional[str] = None,
    provider: Optional[str] = None,
    values: Optional[Dict] = None,
    path: Union[str, Path] = None,
    env_vars: Optional[Dict] = None,
    dryrun: bool = False,
):
    """Builds an instance of :class:`Secret`.

    Args:
        name (str, optional): Name to assign the secret resource.
        provider (str, optional): Provider associated with the secret, if any.
        values (Dict, optional): Dictionary of secret key-value pairs.
        path (str, optional): Path where the secret values are held.
        env_vars (Dict , optional): Dictionary mapping secret keys to the corresponding
            environment variable key.
        dryrun (bool, optional): Whether to create in dryrun mode. (Default: False)

    Returns:
        Secret: The resulting Secret object.

    Example:
        >>> rh.secret("in_memory_secret", values={"secret_key": "secret_val"})
        >>> rh.secret("local_secret", path="secrets.json")
        >>> rh.secret("env_secret", values={"access_key": "12345"}, env_vars={"access_key: "ACCESS_KEY"})
    """
    if provider:
        return provider_secret(provider, name, values, path, env_vars, dryrun)
    if name and not any([provider, values, path, env_vars, dryrun]):
        return Secret.from_name(name, dryrun)
    return Secret(name, values, path, env_vars, dryrun)


def cluster_secret(
    host: str = None,
    name: str = None,
    config: Optional[Dict] = None,
    dryrun: bool = False,
):
    pass


def provider_secret(
    provider: str = None,
    name: str = None,
    values: Optional[Dict] = None,
    path: Union[str, Path] = None,
    env_vars: Optional[Dict] = None,
    dryrun: bool = False,
):
    """Builds an instance of :class:`ProviderSecret`.

    Args:
        provider (str): Provider corresponding to the secret. Currently supported options are:
            ["aws", "azure", "huggingface", "lambda", "github", "gcp", "ssh"]
        name (str, optional): Name to assign the resource. If none is provided, resource name defaults to the
            provider name.
        values (Dict, optional): Dictionary mapping of secret keys and values.
        path (str or Path, optional): Path where the secret values are held.
        env_vars (Dict, optional): Dictionary mapping secret keys to the corresponding
            environment variable key.
        dryrun (bool): Whether to creat in dryrun mode. (Default: False)

    Returns:
        ProviderSecret: The resulting provider secret object.

    Example:
        >>> aws_secret = rh.provider("aws")
        >>> lamdba_secret = rh.provider("lambda", values={"api_key": "xxxxx"})
    """
    if not provider:
        if not name:
            raise ValueError("Either name or provider must be provided.")
        if not any([values, path]):
            return Secret.from_name(name)

    secret_class = _get_provider_class(provider)
    if not any([values, path, env_vars]):
        return secret_class(provider=provider)

    return secret_class(
        name=name,
        provider=provider,
        values=values,
        path=path,
        env_vars=env_vars,
        dryrun=dryrun,
    )


def env_secret(
    name: str = None,
    values: Optional[Dict] = None,
    path: Union[str, Path] = None,
    env_vars: List[str] = None,
    dryrun: bool = False,
):
    """Builds an instance of :class:`EnvSecret`.

    Args:
        name (str, optional): Name to assign the secret resource. If not provided, defautls
            to `"env_vars"`.
        values (Dict, optional): Dictionary of secret key-value pairs. The key
        path (str, optional): Path where the secret values are held.
        env_vars (List , optional): Keys corresponding the environment variable keys.
        dryrun (bool, optional): Whether to create in dryrun mode. (Default: False)

    Returns:
        EnvSecret: The resulting env var secret.

    Example:
        >>> rh.env_secret(path="~/.rh/.env")
        >>> rh.env_secret(values={"PYTHONPATH": "usr/bin/conda"})
        >>> rh.env_secret(env_vars=["PYTHONPATH"])
    """
    if name and not any([values, path, env_vars]):
        return EnvSecret.from_name(name, dryrun)
    if (values and env_vars) and set(values.keys()) != set(env_vars):
        raise Exception(
            "`env_vars` should match the `values` keys if both parameters are provided."
        )
    if values:
        env_vars = list(values.keys())
    return EnvSecret(name, values, path, env_vars, dryrun)
