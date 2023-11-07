from runhouse.resources.resource import Resource

from tests.conftest import init_args


@staticmethod
def test_properties(resource):
    print(resource.rns_address)
    assert isinstance(resource, Resource)
    inits = init_args.get(id(resource))
    assert inits.get("name") in [resource.name, resource.rns_address]


@staticmethod
def test_resource_config_for_rns(resource):
    assert isinstance(resource, Resource)
    config_for_rns = resource.config_for_rns
    assert config_for_rns["name"] == resource.rns_address
