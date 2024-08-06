import os
from pathlib import Path

import boto3
import pytest

import runhouse as rh
from runhouse.globals import rns_client

from .test_helpers.lambda_tests.basic_handler_layer import arr_handler

from .test_helpers.lambda_tests.basic_test_handler import lambda_no_args, lambda_sum

CUR_WORK_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_RESOURCES = f"{CUR_WORK_DIR}/test_helpers/lambda_tests"

CRED_PATH = f"{Path.home()}/.aws/credentials"
DEFAULT_REGION = "us-east-1"


@pytest.fixture(scope="session")
def basic_function():
    my_lambda = rh.aws_lambda_fn(
        fn=lambda_sum,
        runtime="python3.9",
        name="test_lambda_create_and_run",
    )
    my_lambda.save()
    try:
        yield my_lambda
    finally:
        my_lambda.teardown()
        if rns_client.exists(my_lambda.rns_address):
            rns_client.delete_configs(my_lambda)


@pytest.fixture(scope="session")
def numpy_function():
    name = "test_lambda_numpy"
    my_lambda = rh.aws_lambda_fn(
        fn=arr_handler,
        runtime="python3.9",
        name=name,
        env={"reqs": ["numpy", "pandas"], "env_vars": None},
    ).save()
    yield my_lambda
    my_lambda.teardown()
    if rns_client.exists(my_lambda.rns_address):
        rns_client.delete_configs(my_lambda)


def test_create_and_run_no_layers(basic_function):
    res = basic_function(3, 4)
    assert res == 7
    reload_func = rh.aws_lambda_fn(name=basic_function.name)
    res2 = reload_func(12, 7)
    assert res2 == 19
    reload_func_rns_name = rh.aws_lambda_fn(name=basic_function.rns_address)
    res3 = reload_func_rns_name(4, 6)
    assert res3 == 10


def test_load_not_existing_lambda():
    name_no_user = "test_lambda_create_and_run1"
    with pytest.raises(ValueError) as valueError:
        rh.aws_lambda_fn(name=name_no_user)
    assert str(valueError.value) == f"Could not find a Lambda called {name_no_user}."

    name_with_user = "/sashab/test_lambda_no_such_lambda"
    with pytest.raises(ValueError) as valueError:
        rh.aws_lambda_fn(name=name_with_user)
    assert str(valueError.value) == (
        f"Could not find a Lambda called {name_with_user}."
    )


def test_crate_no_arguments():
    with pytest.raises(RuntimeError) as no_args:
        rh.aws_lambda_fn()
    assert str(no_args.value) == (
        "Please provide a callable function OR use from_handler_file method"
        + "in order to create a Lambda function."
    )


def test_bad_runtime_to_factory():
    name = "test_wrong_runtime"

    wrong_runtime_1 = rh.aws_lambda_fn(fn=lambda_sum, runtime=None, name=f"{name}_1")

    wrong_runtime_2 = rh.aws_lambda_fn(fn=lambda_sum, name=f"{name}_2")

    wrong_runtime_3 = rh.aws_lambda_fn(
        fn=lambda_sum, runtime="python3.91", name=f"{name}_3"
    )

    assert wrong_runtime_1.runtime == "python3.9"
    assert wrong_runtime_2.runtime == "python3.9"
    assert wrong_runtime_3.runtime == "python3.9"

    assert wrong_runtime_1.teardown() is True
    if rns_client.exists(wrong_runtime_1.rns_address):
        rns_client.delete_configs(wrong_runtime_1)

    assert wrong_runtime_2.teardown() is True
    if rns_client.exists(wrong_runtime_2.rns_address):
        rns_client.delete_configs(wrong_runtime_1)

    assert wrong_runtime_3.teardown() is True
    if rns_client.exists(wrong_runtime_3.rns_address):
        rns_client.delete_configs(wrong_runtime_3)


def test_func_no_args():
    name = "test_lambda_no_args"
    my_lambda = rh.aws_lambda_fn(
        fn=lambda_no_args,
        runtime="python3.9",
        name=name,
    )
    assert my_lambda() == "no args lambda"
    assert my_lambda.name == "test_lambda_no_args"
    assert my_lambda.teardown() is True


def test_create_and_run_generate_name():
    my_lambda = rh.aws_lambda_fn(fn=lambda_sum, runtime="python3.9")
    res = my_lambda(3, 4)
    assert res == 7
    my_lambda.save()
    reload_func = rh.aws_lambda_fn(name="lambda_sum")
    res2 = reload_func(12, 7)
    assert res2 == 19
    assert my_lambda.teardown() is True
    assert reload_func.teardown() is True


def test_create_and_run_layers_dict(numpy_function):
    res = numpy_function([1, 2, 3], [1, 2, 3])
    assert res == 12


def test_reload_func_with_libs(numpy_function):
    # tests that after the libs are installed, they are not being re-installed.
    my_reloaded_lambda = rh.aws_lambda_fn(name=numpy_function.name)
    res = my_reloaded_lambda([1, 2, 3], [12, 5, 9])
    assert res == 32


def test_create_and_run_layers_env():
    name = "test_lambda_numpy_env"
    my_env = rh.env(reqs=["numpy"])
    my_lambda = rh.aws_lambda_fn(
        fn=arr_handler,
        runtime="python3.9",
        name=name,
        env=my_env,
    )
    res = my_lambda([1, 2, 3], [2, 5, 6])
    assert res == 19
    assert my_lambda.teardown() is True


def test_create_and_run_layers_list():
    name = "test_lambda_numpy_list"
    my_lambda = rh.aws_lambda_fn(
        fn=arr_handler,
        runtime="python3.9",
        name=name,
        env=["numpy"],
    )
    res = my_lambda([1, 2, 3], [4, 7, 9])
    assert res == 26
    assert my_lambda.teardown() is True


def test_layers_increase_timeout_and_memory():
    name = "test_lambda_numpy_increase_params"
    my_lambda = rh.aws_lambda_fn(
        fn=arr_handler,
        runtime="python3.9",
        name=name,
        env=["numpy"],
        timeout=3,
        memory_size=128,
    )
    res = my_lambda([1, 2, 3], [4, 7, 9])
    assert res == 26
    lambda_client = boto3.client("lambda")
    lambda_config = lambda_client.get_function(FunctionName=my_lambda.name)
    assert lambda_config["Configuration"]["Timeout"] == 600
    assert lambda_config["Configuration"]["MemorySize"] == 10240
    assert lambda_config["Configuration"]["EphemeralStorage"]["Size"] == 10240
    assert lambda_config["Configuration"]["FunctionName"] == my_lambda.name
    assert my_lambda.teardown() is True


def test_create_and_run_layers_txt():
    name = "test_lambda_numpy_txt"
    my_lambda = rh.aws_lambda_fn(
        fn=arr_handler,
        runtime="python3.9",
        name=name,
        env=[f"reqs:{TEST_RESOURCES}/"],
    )
    res = my_lambda([1, 2, 3], [1, 2, 3])
    assert res == 12
    assert my_lambda.teardown() is True


def test_update_lambda_one_file():
    name = "test_lambda_create_and_run"
    my_lambda = rh.aws_lambda_fn(fn=lambda_sum, runtime="python3.9", name=name)
    res = my_lambda(6, 4)
    assert res == 10
    reload_func = rh.aws_lambda_fn(name=name)
    res2 = reload_func(12, 13)
    assert res2 == 25


def test_mult_files_each():
    """The handler function calls function from each file separately.
    For example, there are a.py, b.py and c.py. Each file has the following funcs, respectively: func_a, func_b and
    func_c. So in the test, the main function (handler) will look something like this:
    import a.py, b.py and c.py
    def handler_func:
        func_a()
        func_b()
        func_c()
    Also tests the From_handler_file method.
    """
    prefix = "call_files_separately"
    folder_path = f"{TEST_RESOURCES}/{prefix}"
    handler_paths = os.listdir(folder_path)
    handler_paths = [p for p in handler_paths if ".py" in p]
    handler_paths.sort()
    handler_paths = [f"{folder_path}/{p}" for p in handler_paths]
    name = "test_lambda_multiple_files_s"
    my_lambda_calc_1 = rh.LambdaFunction.from_handler_file(
        paths_to_code=handler_paths,
        handler_function_name="my_calc",
        runtime="python3.9",
        name=name,
        env={"reqs": ["numpy"], "env_vars": None},
    )
    res1 = my_lambda_calc_1(2, 3)
    res2 = my_lambda_calc_1(5, 3)
    res3 = my_lambda_calc_1(2, 7)
    res4 = my_lambda_calc_1(10, 5)
    assert res1 == 2.5
    assert res2 == 3.2
    assert res3 == 22.5
    assert res4 == 7.5
    assert my_lambda_calc_1.teardown() is True


def test_few_python_files_chain():
    """The handler function calls functions from different files in chain.
    For example, there are a.py, b.py and c.py. Each file has the following funcs, respectively: func_a, func_b and
    func_c. So in the test, the main function (handler) will look something like this:
    import c.py
    def handler_func:
        func_c()
    where func_c() calls func_b() which calls func_a().
    Also tests the From_handler_file method.
    """
    prefix = "call_files_chain"
    folder_path = f"{TEST_RESOURCES}/{prefix}"
    handler_paths = os.listdir(folder_path)
    handler_paths = [p for p in handler_paths if ".py" in p]
    handler_paths.sort()
    handler_paths = [f"{folder_path}/{p}" for p in handler_paths]
    name = "test_lambda_multiple_files_c"
    my_lambda_calc_2 = rh.LambdaFunction.from_handler_file(
        paths_to_code=handler_paths,
        handler_function_name="special_calc",
        runtime="python3.9",
        name=name,
    )
    res1 = my_lambda_calc_2(2, 3)
    res2 = my_lambda_calc_2(5, 3)
    res3 = my_lambda_calc_2(2, 7)
    res4 = my_lambda_calc_2(10, 5)
    assert res1 == 16
    assert res2 == 17
    assert res3 == 20
    assert res4 == 20
    assert my_lambda_calc_2.teardown() is True


def test_args(basic_function):
    res1 = basic_function(2, 3)
    res2 = basic_function(5, arg2=3)
    res3 = basic_function(arg1=2, arg2=7)
    assert res1 == 5
    assert res2 == 8
    assert res3 == 9


def test_map_starmap(basic_function):
    res_map1 = basic_function.map([1, 2, 3], [4, 5, 6])
    res_map2 = basic_function.map([6, 2, 3], [15, 52, 61])
    res_map3 = basic_function.starmap([(1, 2), (3, 4), (5, 6)])
    res_map4 = basic_function.starmap([(12, 5), (44, 32), (8, 3)])
    assert res_map1 == [5, 7, 9]
    assert res_map2 == [21, 54, 64]
    assert res_map3 == [3, 7, 11]
    assert res_map4 == [17, 76, 11]


def test_create_from_config():
    handler_path = [f"{TEST_RESOURCES}/basic_test_handler.py"]
    name = "test_lambda_from_config"
    config = {
        "paths_to_code": handler_path,
        "handler_function_name": "lambda_sum",
        "runtime": "python3.9",
        "args_names": ["arg1", "arg2"],
        "name": name,
        "tmp_size": 3072,
    }
    config_lambda = rh.LambdaFunction.from_config(config)
    res1 = config_lambda(1, 2)
    res2 = config_lambda(8, 12)
    res3 = config_lambda(14, 17)

    assert res1 == 3
    assert res2 == 20
    assert res3 == 31
    assert config_lambda.teardown() is True


def test_delete_lambda():
    lambda_client = boto3.client("lambda")
    iam_client = boto3.client("iam")
    logs_client = boto3.client("logs")
    name = "test_lambda_to_delete"
    lambda_to_delete = rh.aws_lambda_fn(fn=lambda_sum, runtime="python3.9", name=name)

    lambda_to_delete.save()
    assert lambda_to_delete(5, 11) == 16

    lambda_name = lambda_to_delete.name
    lambda_policy = f"{lambda_name}_Policy"
    lambda_role = f"{lambda_name}_Role"
    lambda_log_group = f"/aws/lambda/{lambda_name}"

    del_res = lambda_to_delete.teardown()
    assert del_res is True

    functions_in_aws = [
        function["FunctionName"]
        for function in lambda_client.list_functions()["Functions"]
    ]
    assert lambda_name not in functions_in_aws

    customer_managed_policies = [
        policy["PolicyName"]
        for policy in iam_client.list_policies(Scope="Local")["Policies"]
    ]
    assert lambda_policy not in customer_managed_policies

    roles_in_aws = [role["RoleName"] for role in iam_client.list_roles()["Roles"]]
    assert lambda_role not in roles_in_aws

    logs_groups_in_aws = [
        log_group["logGroupName"]
        for log_group in logs_client.describe_log_groups()["logGroups"]
    ]
    assert lambda_log_group not in logs_groups_in_aws

    if rns_client.exists(lambda_to_delete.rns_address):
        rns_client.delete_configs(lambda_to_delete)
