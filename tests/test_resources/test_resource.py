import json

import pytest
import runhouse as rh

from tests.conftest import init_args
from tests.utils import friend_account


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
                if resource.rns_address:
                    assert resource.rns_address.split("/")[-1] == args["name"]

        if "dryrun" in args:
            assert args["dryrun"] == resource.dryrun

        assert resource.RESOURCE_TYPE is not None

    @pytest.mark.level("unit")
    def test_config_for_rns(self, resource):
        args = init_args.get(id(resource))
        config = resource.config_for_rns
        assert isinstance(config, dict)
        if resource.rns_address:
            assert config["name"] == resource.rns_address
        else:
            assert config["name"] == resource.name
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
    def test_save_and_load(self, saved_resource):
        # Test loading from name
        loaded_resource = saved_resource.__class__.from_name(saved_resource.rns_address)
        assert loaded_resource.config_for_rns == saved_resource.config_for_rns
        # Changing the name doesn't work for OnDemandCluster, because the name won't match the local sky db
        if isinstance(saved_resource, rh.OnDemandCluster):
            return

        # Do everything inside a try/finally so we don't leave resources behind if the test fails
        try:
            # Test saving under new name
            original_name = saved_resource.rns_address
            alt_name = saved_resource.rns_address + "-alt"

            # This saves a new RNS config with the same resource,
            # but an alt name. It also updates the local config to point to the new RNS config.
            saved_resource.save(alt_name)
            alt_resource = saved_resource.__class__.from_name(alt_name)
            assert alt_resource.config_for_rns == saved_resource.config_for_rns

            # Test that original resource is still available in Den
            reloaded_resource = saved_resource.__class__.from_name(original_name)
            assert reloaded_resource.rns_address != alt_resource.rns_address

            # Rename saved resource locally back to original name
            saved_resource.name = original_name
            assert (
                reloaded_resource.rns_address
                == saved_resource.rns_address
                == original_name
            )

        finally:
            alt_resource.delete_configs()
            assert not rh.exists(alt_resource.rns_address)

    @pytest.mark.level("unit")
    def test_history(self, saved_resource):
        if saved_resource.rns_address[:2] == "~/":
            with pytest.raises(ValueError):
                saved_resource.history()
            return

        history = saved_resource.history()
        assert isinstance(history, list)
        assert isinstance(history[0], dict)
        assert "timestamp" in history[0]
        assert "owner" in history[0]
        assert "data" in history[0]
        # Not all config_for_rns values are saved inside data field
        config = json.loads(
            json.dumps(saved_resource.config_for_rns)
        )  # To deal with tuples and non-json types
        for k, v in history[0]["data"].items():
            if k == "client_port":
                # TODO seems like multiple tests in CI are saving the same resource at the same time, save needs
                # to return a version
                continue
            assert config[k] == v

    @pytest.mark.level("local")
    def test_sharing(
        self, saved_resource, friend_account_logged_in_docker_cluster_pk_ssh
    ):
        # Skip this test for ondemand clusters, because making
        # it compatible with ondemand_cluster requires changes
        # that break CI.
        # TODO: Remove this by doing some CI-specific logic.
        if saved_resource.__class__.__name__ == "OnDemandCluster":
            return

        if saved_resource.rns_address.startswith("~"):
            # For `local_named_resource` resolve the rns address so it can be shared and loaded
            from runhouse.globals import rns_client

            saved_resource.rns_address = rns_client.local_to_remote_address(
                saved_resource.rns_address
            )

        saved_resource.share(
            users=["info@run.house"],
            access_level="read",
            notify_users=False,
        )

        # First try loading in same process/filesystem because it's more debuggable, but not as thorough
        resource_class_name = saved_resource.config_for_rns[
            "resource_type"
        ].capitalize()
        config = saved_resource.config_for_rns
        config.pop(
            "live_state", None
        )  # For ondemand_cluster: too many little differences, leads to flaky tests

        with friend_account():
            assert (
                load_shared_resource_config(
                    resource_class_name, saved_resource.rns_address
                )
                == config
            )

        # TODO: If we are testing with an ondemand_cluster we to
        # sync sky key so loading ondemand_cluster from config works
        # Also need aws secret to load availability zones
        # secrets=["sky", "aws"],
        load_shared_resource_config_cluster = rh.function(
            load_shared_resource_config
        ).to(friend_account_logged_in_docker_cluster_pk_ssh)
        assert (
            load_shared_resource_config_cluster(
                resource_class_name, saved_resource.rns_address
            )
            == config
        )

    # TODO API to run this on local_docker_slim when level == "local"
    @pytest.mark.skip
    def test_loading_in_new_fs(self, resource):
        pass
