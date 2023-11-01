from typing import Dict, Optional

from runhouse.resources.secrets.secret import Secret


def secret(
    name: Optional[str] = None,
    values: Optional[Dict] = None,
    dryrun: bool = False,
):
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
    if name and not values:
        return Secret.from_name(name, dryrun)

    if not values:
        raise ValueError("values must be provided for an in-memory secret.")

    return Secret(name, values, dryrun)
