import pytest

import runhouse as rh

from tests.conftest import init_args


######## Constants ########


######## Fixtures ########


@pytest.fixture(scope="session")
def on_demand_cluster(request):
    return request.getfixturevalue(request.param)


@pytest.fixture(scope="session")
def kubernetes_cpu_cluster():
    c = rh.kubernetes_cluster(
        name="rh-cpu-k8s-test",
        instance_type="1CPU--1GB",
    )

    c.up_if_not()

    # Save to RNS - to be loaded in other tests (ex: Runs)
    c.save()

    # Call save before installing in the event we want to use TLS / den auth
    c.install_packages(["pytest"])
    return c
