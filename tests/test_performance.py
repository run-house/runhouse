import logging
import time
import unittest

import pytest
import requests


logger = logging.getLogger(__name__)


def profile(func, reps=10):
    times = []
    for _ in range(reps):
        start = time.time()
        assert func()
        times.append(round((time.time() - start) * 1000, 2))
    return times, sum(times) / len(times)


@pytest.mark.clustertest
@pytest.mark.rnstest
def test_roundtrip_performance(summer_func):

    times_list, avg_time = profile(lambda: summer_func.system.keys() is not None)
    print(f"Listing keys took {round(avg_time, 2)} ms: {times_list}")

    times_list, avg_time = profile(lambda: summer_func(1, 5) == 6)
    print(f"Call with logs took {round(avg_time, 2)} ms: {times_list}")

    times_list, avg_time = profile(lambda: summer_func(1, 5, stream_logs=False) == 6)
    print(f"Call without logs took {round(avg_time, 2)} ms: {times_list}")

    port = summer_func.system.client.port
    call_url = f"http://127.0.0.1:{port}/call/summer_func/call/?serialization=None"
    times_list, avg_time = profile(
        lambda: requests.post(call_url, json={"args": [1, 2]}).json() == 3
    )
    print(f"HTTP call took {round(avg_time, 2)} ms: {times_list}")


@pytest.mark.clustertest
@pytest.mark.rnstest
@unittest.skip("Not implemented yet.")
def test_https_performance(summer_func):
    pass


@pytest.mark.clustertest
@pytest.mark.rnstest
@unittest.skip("Not implemented yet.")
def test_https_with_den_auth_performance(summer_func):
    pass


@pytest.mark.clustertest
@pytest.mark.rnstest
@unittest.skip("Not implemented yet.")
def test_http_performance(summer_func):
    pass


@pytest.mark.clustertest
@pytest.mark.rnstest
@unittest.skip("Not implemented yet.")
def test_ssh_performance(summer_func):
    pass


if __name__ == "__main__":
    unittest.main()
