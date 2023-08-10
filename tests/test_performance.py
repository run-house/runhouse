import logging
import unittest
import time

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

    times_list, avg_time = profile(lambda: summer_func.system.list_keys() is not None)
    print(f"list_keys took {round(avg_time, 2)} ms: {times_list}")

    times_list, avg_time = profile(lambda: summer_func(1, 5) == 6)
    print(f"Call with logs took {round(avg_time, 2)} ms: {times_list}")

    times_list, avg_time = profile(lambda: summer_func(1, 5, stream_logs=False) == 6)
    print(f"Call without logs took {round(avg_time, 2)} ms: {times_list}")

    call_url = "http://127.0.0.1:50052/call/summer_func/call/?serialization=None"
    times_list, avg_time = profile(lambda: requests.post(call_url, json={"args": [1, 2]}).json() == 3)
    print(f"HTTP call took {round(avg_time, 2)} ms: {times_list}")

if __name__ == "__main__":
    unittest.main()