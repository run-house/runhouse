import pytest

import runhouse as rh

from tests.conftest import init_args


@pytest.fixture
def env(request):
    """Parametrize over multiple envs - useful for running the same test on multiple envs."""
    return request.getfixturevalue(request.param.__name__)


@pytest.fixture
def test_env():
    args = {"reqs": ["pytest"]}
    e = rh.env(**args)
    init_args[id(e)] = args
    return e
