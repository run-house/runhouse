import os

TEST_ORG = "test-org"
TESTING_LOG_LEVEL = "DEBUG"
TESTING_AUTOSTOP_INTERVAL = 15

TEST_ENV_VARS = {
    "var1": "val1",
    "var2": "val2",
    "RH_LOG_LEVEL": os.getenv("RH_LOG_LEVEL") or TESTING_LOG_LEVEL,
    "RH_AUTOSTOP_INTERVAL": str(
        os.getenv("RH_AUTOSTOP_INTERVAL") or TESTING_AUTOSTOP_INTERVAL
    ),
}

TEST_REQS = ["pytest", "httpx", "pytest_asyncio", "pandas", "numpy<=1.26.4"]
