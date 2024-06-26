import pytest
import requests
import runhouse as rh

from tests.test_resources.test_modules.test_server_modules.assets.sample_fastapi_app import (
    app,
)


"""
Resources:
https://github.com/simonw/asgi-proxy-lib
https://github.com/valohai/asgiproxy
https://github.com/florimondmanca/awesome-asgi

TODO test with Den Auth and HTTPS enabled
"""


@pytest.mark.level("local")
async def test_asgi_server(cluster):
    fast_api_module = rh.asgi(app).to(
        cluster, env=["pytest", "requests"], name="fast_api_module"
    )
    assert isinstance(fast_api_module, rh.Asgi)
    assert fast_api_module.summer(1, 2) == 3

    # Call an async method
    assert await fast_api_module.my_deeply_nested_async_endpoint("hello", 1, 2.0) == (
        "hello",
        1,
        2.0,
    )

    assert list(fast_api_module.my_streaming_endpoint()) == list(range(10))
    assert fast_api_module.my_endpoint_with_optional_body_params_and_header(
        a=1, b=2, c=3
    ) == (1, 2, 3)

    endpoint = fast_api_module.endpoint(external=False)
    res = requests.get(f"{endpoint}/summer/1", params={"b": 2})
    assert res.json() == 3
    res = requests.post(
        f"{endpoint}/my/deeply/hello/nested/endpoint/1",
        json={"arg3": 2.0},
    )
    assert res.json() == ["hello", 1, 2.0]
    res = requests.get(f"{endpoint}/my/streaming/endpoint")
    assert res.json() == list(range(10))
    res = requests.get(
        f"{endpoint}/my/endpoint/with/optional/body/params/and/header",
        json={"a": 1, "b": 2},
        headers={"c": 3},
    )
    assert res.json() == [1, 2, 3]

    # Test UI page pulls properly
    res = requests.get(f"{endpoint}/docs")
    assert res.status_code == 200


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
