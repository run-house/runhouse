from pathlib import Path
from typing import Dict, List, Optional, Union

from runhouse.resources.blobs.file import File
from runhouse.resources.secrets.env_var_secret import EnvVarSecret
from runhouse.resources.secrets.provider_secrets.provider_secret import ProviderSecret
from runhouse.resources.secrets.secret import Secret

from .provider_secrets.providers import _get_provider_class


def secret(
    name: Optional[str] = None,
    values: Optional[Dict] = None,
    provider: Optional[str] = None,
    dryrun: bool = False,
) -> Secret:
    """Builds an instance of :class:`Secret`.

    Args:
        name (str, optional): Name to assign the secret resource.
        values (Dict, optional): Dictionary of secret key-value pairs.
        dryrun (bool, optional): Whether to create in dryrun mode. (Default: False)
    Returns:
        Secret: The resulting Secret object.

    Example:
        >>> rh.secret("in_memory_secret", values={"secret_key": "secret_val"})
    """
    if provider:
        return provider_secret(
            name=name, provider=provider, values=values, dryrun=dryrun
        )

    if name and not values:
        return Secret.from_name(name, dryrun)

    if not values:
        raise ValueError("values must be provided for an in-memory secret.")

    return Secret(name, values, dryrun)


def provider_secret(
    provider: Optional[str] = None,
    name: Optional[str] = None,
    values: Optional[Dict] = None,
    path: Union[str, File] = None,
    env_vars: Optional[Dict] = None,
    dryrun: bool = False,
) -> ProviderSecret:
    """
    Builds an instance of :class:`ProviderSecret`. At most one of values, path, and env_vars
    can be provided, to maintain one source of truth. If None are provided, will infer the values
    from the default path or env vars for the given provider.

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
        >>> aws_secret = rh.provider_secret("aws")
        >>> lamdba_secret = rh.provider_secret("lambda", values={"api_key": "xxxxx"})
        >>>
    """
    if not provider:
        if not name:
            raise ValueError("Either name or provider must be provided.")
        if not any([values, path, env_vars]):
            return Secret.from_name(name)

    elif not any([values, path, env_vars]):
        secret_class = _get_provider_class(provider)
        return secret_class(name=name, provider=provider, dryrun=dryrun)

    elif sum([bool(x) for x in [values, path, env_vars]]) == 1:
        secret_class = _get_provider_class(provider)
        return secret_class(
            name=name,
            provider=provider,
            values=values,
            path=path,
            env_vars=env_vars,
            dryrun=dryrun,
        )

    raise ValueError("Only one of values, path, and env_vars should be set.")


def env_var_secret(
    name: str = None,
    env_vars: List[str] = None,
    values: Optional[Dict] = None,
    path: Union[str, Path] = None,
    dryrun: bool = False,
) -> EnvVarSecret:
    """
    Builds an instance of :class:`EnvVarSecret`. At most one of of env_vars and values
    can be provided, to maintain one source of truth.

    Args:
        name (str, optional): Name to assign the resource. If none is provided, resource name defaults to the
            provider name.
        env_vars (List[str], optional): Dictionary mapping secret keys to the corresponding
            environment variable key.
        values (Dict, optional): Dictionary mapping of environment variable keys to values.
        dryrun (bool): Whether to creat in dryrun mode. (Default: False)

    Returns:
        EnvVarSecret: The resulting env var secret object.

    Example:
        >>> local_env_var_secret = rh.env_var_secret(env_vars=["PYTHONPATH"])
        >>> env_secret = rh.env_var_secret(values={"API_KEY": "abcd"})
    """
    if name and not any([env_vars, values, path]):
        return EnvVarSecret.from_name(name, dryrun)
    if env_vars and values:
        raise ValueError(
            "Only one of env_vars and values should be set. One will be infered from the other."
        )

    return EnvVarSecret(
        name=name,
        env_vars=env_vars,
        values=values,
        path=path,
        dryrun=dryrun,
    )
