import pytest

import runhouse as rh

from ....conftest import init_args

######## Constants ########


######## Fixtures ########


@pytest.fixture(scope="session")
def on_demand_cluster(request):
    return request.getfixturevalue(request.param.__name__)


@pytest.fixture(
    params=[
        "ondemand_cpu_cluster",
        "v100_gpu_cluster",
        "k80_gpu_cluster",
        "a10g_gpu_cluster",
    ],
    ids=["cpu", "v100", "k80", "a10g"],
)
def ondemand_cluster(request):
    return request.getfixturevalue(request.param)


@pytest.fixture(scope="session")
def ondemand_cpu_cluster():
    args = {"name": "^rh-cpu"}
    c = rh.ondemand_cluster(**args)
    init_args[id(c)] = args

    c.up_if_not()

    # Save to RNS - to be loaded in other tests (ex: Runs)
    c.save()

    # Call save before installing in the event we want to use TLS / den auth
    c.install_packages(["pytest"])
    return c


@pytest.fixture(scope="session")
def v100_gpu_cluster():
    return rh.ondemand_cluster(
        name="rh-v100", instance_type="V100:1", provider="aws"
    ).up_if_not()


@pytest.fixture(scope="session")
def k80_gpu_cluster():
    return rh.ondemand_cluster(
        name="rh-k80", instance_type="K80:1", provider="aws"
    ).up_if_not()


@pytest.fixture(scope="session")
def a10g_gpu_cluster():
    return rh.ondemand_cluster(
        name="rh-a10x", instance_type="g5.2xlarge", provider="aws"
    ).up_if_not()


@pytest.fixture(scope="session")
def ondemand_https_cluster_with_auth():
    c = rh.ondemand_cluster(
        name="rh-cpu-https",
        instance_type="CPU:2+",
        den_auth=True,
        server_connection_type="tls",
        open_ports=[443],
    )
    c.up_if_not()

    c.install_packages(["pytest"])
    return c
