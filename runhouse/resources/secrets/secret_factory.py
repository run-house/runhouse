from pathlib import Path
from typing import Dict, List, Optional, Union

from runhouse.resources.secrets.env_secret import EnvSecret
from runhouse.resources.secrets.provider_secrets.providers import _get_provider_class

from runhouse.resources.secrets.secret import Secret


def secret(
    name: Optional[str] = None,
    provider: Optional[str] = None,
    secrets: Optional[Dict] = None,
    path: Union[str, Path] = None,
    env_vars: Optional[Dict] = None,
    dryrun: bool = False,
):
    """Builds an instance of :class:`Secret`.

    Args:
        name (str, optional): Name to assign the secret resource.
        provider (str, optional): Provider associated with the secret, if any.
        secrets (Dict, optional): Dictionary of secret key-value pairs.
        path (str, optional): Path where the secrets are held.
        env_vars (Dict , optional): Dictionary mapping secrets keys to the corresponding
            environment variable key.
        dryrun (bool, optional): Whether to create in dryrun mode. (Default: False)

    Returns:
        Secret: The resulting Secret object.

    Example:
        >>> rh.secret("in_memory_secret", secrets={"secret_key": "secret_val"})
        >>> rh.secret("local_secret", path="secrets.json")
        >>> rh.secret("env_secret", secrets={"access_key": "12345"}, env_vars={"access_key: "ACCESS_KEY"})
    """
    if not (name or provider):
        raise ValueError(
            "Either name or provider must be provided."
        )  # TODO: not necessarily if env vars
    if provider:
        return provider_secret(provider, name, secrets, path, env_vars, dryrun)
    if name and not any([provider, secrets, path, env_vars, dryrun]):
        return Secret.from_name(name, dryrun)
    return Secret(name, secrets, path, env_vars, dryrun)


def cluster_secret():
    pass


def provider_secret(
    provider,
    name: str = None,
    secrets: Optional[Dict] = None,
    path: Union[str, Path] = None,
    env_vars: Optional[Dict] = None,
    dryrun: bool = False,
):
    """Builds an instance of :class:`ProviderSecret`.

    Args:
        provider (str): Provider corresponding to the secret.
            Currently supported options are: ["aws", "gcp", "lambda"]
        name (str, optional): Name to assign the resource. If none is provided, resource name defaults to the
            provider name.
        secrets (Dict, optional): Dictionary mapping of secrets keys and values.
        path (str or Path, optional): Path where the secrets are held.
        env_vars (Dict, optional): Dictionary mapping secrets keys to the corresponding
            environment variable key.
        dryrun (bool): Whether to creat in dryrun mode. (Default: False)

    Returns:
        ProviderSecret: The resulting provider secret object.

    Example:
        >>> aws_secret = rh.provider("aws")
        >>> lamdba_secret = rh.provider("lambda", secrets={"api_key": "xxxxx"})
    """
    secret_class = _get_provider_class(provider)

    if not any([secrets, path]):
        return (
            secret_class.from_name(name) if name else secret_class.from_name(provider)
        )
    return secret_class(
        name=name,
        provider=provider,
        secrets=secrets,
        path=path,
        env_vars=env_vars,
        dryrun=dryrun,
    )


def env_secret(
    name: str = None,
    secrets: Optional[Dict] = None,
    path: Union[str, Path] = None,
    env_vars: List[str] = None,
    dryrun: bool = False,
):
    """Builds an instance of :class:`EnvSecret`.

    Args:
        name (str, optional): Name to assign the secret resource. If not provided, defautls
            to `"env_vars"`.
        secrets (Dict, optional): Dictionary of secret key-value pairs. The key
        path (str, optional): Path where the secrets are held.
        env_vars (List , optional): Keys corresponding the environment variable keys.
        dryrun (bool, optional): Whether to create in dryrun mode. (Default: False)

    Returns:
        EnvSecret: The resulting env var secret.

    Example:
        >>> rh.env_secret(path="~/.rh/.env")
        >>> rh.env_secret(secrets={"PYTHONPATH": "usr/bin/conda"})
        >>> rh.env_secret(env_vars=["PYTHONPATH"])
    """
    if name and not any([secrets, path, env_vars]):
        return EnvSecret.from_name(name, dryrun)
    if (secrets and env_vars) and set(secrets.keys()) != set(env_vars):
        raise Exception(
            "`env_vars` should match the `secrets` keys if both parameters are provided."
        )
    if secrets:
        env_vars = list(secrets.keys())
    return EnvSecret(name, secrets, path, env_vars, dryrun)
