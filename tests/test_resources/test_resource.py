import unittest

import pytest
import runhouse as rh

from tests.test_resources.conftest import (
    local_named_resource,
    named_resource,
    unnamed_resource,
)

from tests.conftest import init_args


class TestResource:

    UNIT = {"resource": [unnamed_resource, named_resource, local_named_resource]}
    LOCAL = {"resource": [unnamed_resource, named_resource, local_named_resource]}
    MINIMAL = {"resource": [named_resource]}
    FULL = {"resource": [unnamed_resource, named_resource, local_named_resource]}
    ALL = {"resource": [unnamed_resource, named_resource, local_named_resource]}

    def test_resource_factory_and_properties(self, resource):
        assert isinstance(resource, rh.Resource)
        args = init_args.get(id(resource))
        if "name" in args:
            if "/" in args["name"]:
                assert resource.rns_address == args["name"]
            else:
                assert args["name"] == resource.name
                assert resource.rns_address.split("/")[-1] == args["name"]

        if "dryrun" in args:
            assert args["dryrun"] == resource.dryrun

        assert resource.RESOURCE_TYPE is not None

    def test_config_for_rns(self, resource):
        args = init_args.get(id(resource))
        config = resource.config_for_rns
        assert isinstance(config, dict)
        assert config["name"] == resource.rns_address
        assert config["resource_type"] == resource.RESOURCE_TYPE
        assert config["resource_subtype"] == resource.__class__.__name__
        if "dryrun" in args:
            assert config["dryrun"] == args["dryrun"]

    def test_from_config(self, resource):
        config = resource.config_for_rns
        new_resource = rh.Resource.from_config(config)
        assert new_resource.config_for_rns == resource.config_for_rns
        assert new_resource.rns_address == resource.rns_address
        assert new_resource.dryrun == resource.dryrun
        # TODO allow resource subclass tests to extend set of properties to test

    def test_save_and_load(self, resource):
        if resource.name is None:
            with pytest.raises(ValueError):
                resource.save()
            assert resource.name is None
            return

        # Test savng and then loading from name
        resource.save()
        loaded_resource = resource.__class__.from_name(resource.rns_address)
        assert loaded_resource.config_for_rns == resource.config_for_rns

        # Test saving under new name
        original_name = resource.rns_address
        alt_name = resource.rns_address + "_alt"
        resource.save(alt_name)
        loaded_resource = resource.__class__.from_name(alt_name)
        assert loaded_resource.config_for_rns == resource.config_for_rns
        loaded_resource.delete_configs()

        # Test that original resource is still available
        resource = resource.__class__.from_name(original_name)
        assert resource.rns_address != loaded_resource.rns_address

        resource.delete_configs()
        assert not rh.exists(resource.rns_address)

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
        assert "creator" in history[0]
        assert "data" in history[0]
        # Not all config_for_rns values are saved inside data field
        for k,v in history[0]["data"].items():
            assert resource.config_for_rns[k] == v
        resource.delete_configs()

    # TODO API to run this on local_docker_slim when level == "local"
    @pytest.mark.skip
    def test_loading_in_new_fs(self, resource):
        pass


if __name__ == "__main__":
    unittest.main()
