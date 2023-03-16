import os
import runhouse as rh
from runhouse.rh_config import configs


def test_login_flow_in_new_env():
    token = os.getenv('TEST_TOKEN')
    configs.set('token', token)  # Set in config to be used in requests to RNS

    secrets_in_vault = rh.Secrets.download_into_env(save_locally=False, check_enabled=False)
    assert secrets_in_vault, "No secrets found in Vault"

    providers_in_vault = list(secrets_in_vault)

    # Download secrets from vault into docker env
    rh.login(token=token, download_secrets=True)

    enabled_providers = rh.Secrets.enabled_providers(as_str=True)

    for provider in providers_in_vault:
        provider_config = configs.get("secrets", {}).get(provider)
        if provider_config is None and provider in enabled_providers:
            assert False, f'Provider {provider} is enabled but missing from rh config'

    assert True
