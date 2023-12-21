import json
import unittest

import pytest
import runhouse as rh

from tests.conftest import init_args
from tests.utils import test_account


def load_shared_resource_config(resource_class_name, address):
    resource_class = getattr(rh, resource_class_name)
    loaded_resource = resource_class.from_name(address, dryrun=True)
    config = loaded_resource.config_for_rns
    config.pop("live_state", None)  # Too many little differences, leads to flaky tests
    return config
    # TODO allow resource subclass tests to extend set of properties to test


class TestResource:

    UNIT = {"resource": ["unnamed_resource", "named_resource", "local_named_resource"]}
    LOCAL = {"resource": ["unnamed_resource", "named_resource", "local_named_resource"]}
    MINIMAL = {"resource": ["named_resource"]}
    THOROUGH = {
        "resource": ["unnamed_resource", "named_resource", "local_named_resource"]
    }
    MAXIMAL = {
        "resource": ["unnamed_resource", "named_resource", "local_named_resource"]
    }

    @pytest.mark.level("unit")
    def test_resource_factory_and_properties(self, resource):
        assert isinstance(resource, rh.Resource)
        args = init_args.get(id(resource))
        if "name" in args:

            if args["name"].startswith("^"):
                args["name"] = args["name"][1:]

            if "/" in args["name"]:
                assert resource.rns_address == args["name"]
            else:
                assert resource.name == args["name"]
                assert resource.rns_address.split("/")[-1] == args["name"]

        if "dryrun" in args:
            assert args["dryrun"] == resource.dryrun

        assert resource.RESOURCE_TYPE is not None

    @pytest.mark.level("unit")
    def test_config_for_rns(self, resource):
        args = init_args.get(id(resource))
        config = resource.config_for_rns
        assert isinstance(config, dict)
        assert config["name"] == resource.rns_address
        assert config["resource_type"] == resource.RESOURCE_TYPE
        assert config["resource_subtype"] == resource.__class__.__name__
        if "dryrun" in args:
            assert config["dryrun"] == args["dryrun"]

    @pytest.mark.level("unit")
    def test_from_config(self, resource):
        config = resource.config_for_rns
        new_resource = rh.Resource.from_config(config)
        assert new_resource.config_for_rns == resource.config_for_rns
        assert new_resource.rns_address == resource.rns_address
        assert new_resource.dryrun == resource.dryrun
        # TODO allow resource subclass tests to extend set of properties to test

    @pytest.mark.level("unit")
    def test_save_and_load(self, resource):
        if resource.name is None:
            with pytest.raises(ValueError):
                resource.save()
            assert resource.name is None
            return

        # Test saving and then loading from name
        resource.save()
        loaded_resource = resource.__class__.from_name(resource.rns_address)
        assert loaded_resource.config_for_rns == resource.config_for_rns

        # Changing the name doesn't work for OnDemandCluster, because the name won't match the local sky db
        if isinstance(resource, rh.OnDemandCluster):
            return

        # Test saving under new name
        original_name = resource.rns_address
        try:
            alt_name = resource.rns_address + "-alt"
            resource.save(alt_name)
            loaded_resource = resource.__class__.from_name(alt_name)
            assert loaded_resource.config_for_rns == resource.config_for_rns
            loaded_resource.delete_configs()

            # Test that original resource is still available
            reloaded_resource = resource.__class__.from_name(original_name)
            assert reloaded_resource.rns_address != loaded_resource.rns_address

        finally:
            resource.save(original_name)
            resource.delete_configs()
            assert not rh.exists(resource.rns_address)
            assert not rh.exists(loaded_resource.rns_address)

        # Final check to make sure we didn't mess anything up for subsequent tests
        assert resource.rns_address == original_name

    @pytest.mark.level("unit")
    def test_history(self, resource):
        if resource.name is None or resource.rns_address[:2] == "~/":
            with pytest.raises(ValueError):
                resource.history()
            return

        resource.save()
        history = resource.history()
        assert isinstance(history, list)
        assert isinstance(history[0], dict)
        assert "timestamp" in history[0]
        assert "owner" in history[0]
        assert "data" in history[0]
        # Not all config_for_rns values are saved inside data field
        config = json.loads(
            json.dumps(resource.config_for_rns)
        )  # To deal with tuples and non-json types
        for k, v in history[0]["data"].items():
            assert config[k] == v
        resource.delete_configs()

    @pytest.mark.level("local")
    def test_sharing(self, resource, local_test_account_cluster_public_key):
        if resource.name is None:
            with pytest.raises(ValueError):
                resource.save()
            assert resource.name is None
            return

        if resource.rns_address.startswith("~"):
            # For `local_named_resource` resolve the rns address so it can be shared and loaded
            from runhouse.globals import rns_client

            resource.rns_address = rns_client.local_to_remote_address(
                resource.rns_address
            )

        resource.share(
            users=["info@run.house"],
            access_level="read",
            notify_users=False,
        )

        # First try loading in same process/filesystem because it's more debuggable, but not as thorough
        resource_class_name = resource.__class__.__name__
        config = resource.config_for_rns
        config.pop(
            "live_state", None
        )  # For ondemand_cluster: too many little differences, leads to flaky tests

        with test_account():
            assert (
                load_shared_resource_config(resource_class_name, resource.rns_address)
                == config
            )

        load_shared_resource_config_cluster = rh.function(
            load_shared_resource_config
        ).to(
            system=local_test_account_cluster_public_key,
            env=rh.env(
                working_dir=None,
                # Sync sky key so loading ondemand_cluster from config works
                # Also need aws secret to load availability zones
                secrets=["ssh-sky-key", "aws"],
            ),
        )
        assert (
            load_shared_resource_config_cluster(
                resource_class_name, resource.rns_address
            )
            == config
        )

    # TODO API to run this on local_docker_slim when level == "local"
    @pytest.mark.skip
    def test_loading_in_new_fs(self, resource):
        pass


if __name__ == "__main__":
    unittest.main()
