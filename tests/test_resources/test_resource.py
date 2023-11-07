import unittest

import pytest

from tests.test_resources.conftest import (
    local_named_resource,
    named_resource,
    unnamed_resource,
)


class TestResource:

    UNIT = {"resource": [unnamed_resource, named_resource, local_named_resource]}
    LOCAL = {"resource": [unnamed_resource, local_named_resource]}
    MINIMAL = {"resource": [named_resource]}
    FULL = {"resource": [local_named_resource]}
    ALL = {"resource": [unnamed_resource, named_resource, local_named_resource]}

    def test_save_and_load(self, resource):
        if resource.name is None:
            with pytest.raises(ValueError):
                resource.save()
            assert resource.name is None
            return

        resource.save()
        loaded_resource = resource.__class__.from_name(resource.name)
        assert loaded_resource.config_for_rns == resource.config_for_rns

        original_name = resource.name
        alt_name = resource.name + "_alt"
        resource.save(alt_name)
        loaded_resource = resource.__class__.from_name(alt_name)
        assert loaded_resource.config_for_rns == resource.config_for_rns

        # Test that original resource is still available
        resource = resource.__class__.from_name(original_name)
        resource.save(overwrite=True)
        assert resource.rns_address != loaded_resource.rns_address

    def test_history(self, resource):
        if resource.name is None or resource.rns_address[:2] == "~/":
            with pytest.raises(ValueError):
                resource.history()
            return

        history = resource.history()
        assert isinstance(history, list)
        assert isinstance(history[0], dict)
        assert "timestamp" in history[0]
        assert "version" in history[0]
        assert "config" in history[0]

    # TODO API to run this on local_docker_slim when level == "local"
    @pytest.mark.skip
    def test_loading_in_new_fs(self, resource):
        pass


if __name__ == "__main__":
    unittest.main()
