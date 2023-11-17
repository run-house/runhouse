import asyncio
import logging
import unittest

import pytest
import requests
import runhouse as rh

from tests.test_resources.test_modules.test_server_modules.assets.sample_fastapi_app import (
    app,
)

logger = logging.getLogger(__name__)

"""
Resources:
https://github.com/simonw/asgi-proxy-lib
https://github.com/valohai/asgiproxy
https://github.com/florimondmanca/awesome-asgi

TODO test with Den Auth and HTTPS enabled
"""


def test_asgi_server(cluster):
    # If we detect that this is a FastAPI app / ASGIApp, we can automatically attach the functions to the  __getattr__
    # of the app object, so that we can call them directly from the app object. We can also deploy it with uvicorn.
    # We can find the app name and module via the functions in the app object's routes:
    # app.routes[4].endpoint.__module__
    # or maybe app.routes[4].dependant.cache_key[0]
    # We can find the app name by extracting the decorator from the function source:
    # inspect.getsource(app.routes[4].endpoint)
    fast_api_module = rh.server(app).to(
        cluster, env=["pytest", "requests"], name="fast_api_module"
    )
    assert isinstance(fast_api_module, rh.ASGIApp)
    assert fast_api_module.summer(1, 2) == 3
    # Call an async method
    assert asyncio.run(
        fast_api_module.my_deeply_nested_saync_endpoint("hello", 1, 2.0)
        == ("hello", 1, 2.0)
    )

    assert fast_api_module.my_streaming_endpoint() == list(range(10))
    assert fast_api_module.my_endpoint_with_optional_body_params_and_header(
        a=1, b=2, c=3
    ) == (1, 2, 3)

    res = requests.get(
        "http://localhost:8000/fast_api_module/summer/1", params={"b": 2}
    )
    assert res.json() == 3
    res = requests.post(
        "http://localhost:8000/fast_api_module/my/deeply/hello/nested/endpoint/1",
        json={"arg3": 2.0},
    )
    assert res.json() == ["hello", 1, 2.0]
    res = requests.get("http://localhost:8000/fast_api_module/my/streaming/endpoint")
    assert res.json() == list(range(10))
    res = requests.get(
        "http://localhost:8000/fast_api_module/my/endpoint/with/optional/body/params/and/header",
        json={"a": 1, "b": 2},
        headers={"c": 3},
    )
    assert res.json() == [1, 2, 3]

    # Test UI page pulls properly
    res = requests.get("http://localhost:8000/fast_api_module/docs")
    assert res.status_code == 200

    fast_api_module.stop()

    # Test that the server is stopped
    with pytest.raises(requests.exceptions.ConnectionError):
        requests.get("http://localhost:8000/fast_api_module/summer/1", params={"b": 2})


def test_python_script_start(cluster):
    mod = (
        tests.test_resources.test_modules.test_server_modules.assets.sample_fastapi_app
    )
    # Will start with python -m tests.test_resources.test_modules.test_server_modules.assets.sample_fastapi_app
    server_module = rh.server(mod, port=8000).to(
        cluster, env=["pytest", "requests"], name="fast_api_module"
    )
    assert isinstance(server_module, rh.WebServer)

    res = requests.get(
        "http://localhost:8001/fast_api_module/summer/1", params={"b": 2}
    )
    assert res.json() == 3
    res = requests.post(
        "http://localhost:8001/fast_api_module/my/deeply/hello/nested/endpoint/1",
        json={"arg3": 2.0},
    )
    assert res.json() == ["hello", 1, 2.0]
    res = requests.get("http://localhost:8001/fast_api_module/my/streaming/endpoint")
    assert res.json() == list(range(10))
    res = requests.get(
        "http://localhost:8001/fast_api_module/my/endpoint/with/optional/body/params/and/header",
        json={"a": 1, "b": 2},
        headers={"c": 3},
    )
    assert res.json() == [1, 2, 3]

    # Test UI page pulls properly
    res = requests.get("http://localhost:8000/fast_api_module/docs")
    assert res.status_code == 200

    fast_api_module.stop()

    # Test that the server is stopped
    with pytest.raises(requests.exceptions.ConnectionError):
        requests.get("http://localhost:8001/fast_api_module/summer/1", params={"b": 2})

    server_module.stop()


def test_custom_start_cmd(cluster):
    # Will start with python -m tests.test_resources.test_modules.test_server_modules.assets.sample_fastapi_app
    start_cmd = (
        "uvicorn --port 8001 --host 0.0.0.0 "
        "runhouse.tests.test_resources.test_modules.test_server_modules.assets.sample_fastapi_app:app"
    )
    server_module = rh.server(start_cmd, port=8000).to(
        cluster, env=["pytest", "requests"], name="fast_api_module"
    )
    assert isinstance(server_module, rh.WebServer)

    res = requests.get(
        "http://localhost:8001/fast_api_module/summer/1", params={"b": 2}
    )
    assert res.json() == 3
    res = requests.post(
        "http://localhost:8001/fast_api_module/my/deeply/hello/nested/endpoint/1",
        json={"arg3": 2.0},
    )
    assert res.json() == ["hello", 1, 2.0]
    res = requests.get("http://localhost:8001/fast_api_module/my/streaming/endpoint")
    assert res.json() == list(range(10))
    res = requests.get(
        "http://localhost:8001/fast_api_module/my/endpoint/with/optional/body/params/and/header",
        json={"a": 1, "b": 2},
        headers={"c": 3},
    )

    # Test UI page pulls properly
    res = requests.get("http://localhost:8000/fast_api_module/docs")
    assert res.status_code == 200

    fast_api_module.stop()
    with pytest.raises(requests.exceptions.ConnectionError):
        requests.get("http://localhost:8001/fast_api_module/summer/1", params={"b": 2})

    server_module.stop()


if __name__ == "__main__":
    unittest.main()
