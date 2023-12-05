import logging
import os
import unittest
from pathlib import Path

import boto3
import pytest

import runhouse as rh

logger = logging.getLogger(__name__)
CUR_WORK_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_RESOURCES = f"{CUR_WORK_DIR}/test_helpers/lambda_tests"

CRED_PATH = f"{Path.home()}/.aws/credentials"
DEFAULT_REGION = "us-east-1"

if Path(CRED_PATH).is_file():
    LAMBDA_CLIENT = boto3.client("lambda")
else:
    LAMBDA_CLIENT = boto3.client("lambda", region_name=DEFAULT_REGION)
IAM_CLIENT = boto3.client("iam")
LAMBDAS_NAMES = set()


def test_create_and_run_no_layers():
    handler_path = [f"{TEST_RESOURCES}/basic_test_handler.py"]
    name = "test_lambda_create_and_run"
    my_lambda = rh.aws_lambda_function(
        paths_to_code=handler_path,
        handler_function_name="lambda_sum",
        runtime="python3.9",
        args_names=["arg1", "arg2"],
        name=name,
    )

    my_lambda.save()
    res = my_lambda(3, 4)
    assert res == "7"
    reload_func = rh.aws_lambda_function(name=name)
    res2 = reload_func(12, 7)
    assert res2 == "19"
    LAMBDAS_NAMES.add(my_lambda.name)


def test_load_not_existing_lambda():
    name = "test_lambda_create_and_run1"
    with pytest.raises(RuntimeError) as runtime_error:
        rh.aws_lambda_function(name=name)
    assert str(runtime_error.value) == (
        f"Could not find a Lambda called {name}. Please provide a name of an "
        + "existing Lambda, or paths_to_code, handler_function_name, runtime "
        + "and args_names (and a name if you wish), in order to create a new lambda."
    )


def test_crate_no_arguments():
    with pytest.raises(RuntimeError) as no_args:
        rh.aws_lambda_function()
    assert str(no_args.value) == "Please provide a path to the lambda handler file."


def test_bad_handler_path_to_factory():
    name = "test_lambda_create_and_run"
    with pytest.raises(RuntimeError) as no_handler_path:
        rh.aws_lambda_function(
            handler_function_name="lambda_sum",
            runtime="python3.9",
            args_names=["arg1", "arg2"],
            name=name,
        )
    assert (
        str(no_handler_path.value)
        == "Please provide a path to the lambda handler file."
    )

    with pytest.raises(RuntimeError) as handler_path_none:
        rh.aws_lambda_function(
            paths_to_code=None,
            handler_function_name="lambda_sum",
            runtime="python3.9",
            args_names=["arg1", "arg2"],
            name=name,
        )
    assert (
        str(handler_path_none.value)
        == "Please provide a path to the lambda handler file."
    )

    with pytest.raises(RuntimeError) as handler_path_empty:
        rh.aws_lambda_function(
            paths_to_code=[],
            handler_function_name="lambda_sum",
            runtime="python3.9",
            args_names=["arg1", "arg2"],
            name=name,
        )
    assert (
        str(handler_path_empty.value)
        == "Please provide a path to the lambda handler file."
    )


def test_bad_handler_func_name_to_factory():
    name = "test_lambda_create_and_run"
    handler_path = [f"{TEST_RESOURCES}/basic_test_handler.py"]
    with pytest.raises(RuntimeError) as no_func_name:
        rh.aws_lambda_function(
            paths_to_code=handler_path,
            runtime="python3.9",
            args_names=["arg1", "arg2"],
            name=name,
        )
    assert (
        str(no_func_name.value)
        == "Please provide the name of the function that should be executed by the lambda."
    )

    with pytest.raises(RuntimeError) as func_name_none:
        rh.aws_lambda_function(
            paths_to_code=handler_path,
            handler_function_name=None,
            runtime="python3.9",
            args_names=["arg1", "arg2"],
            name=name,
        )
    assert (
        str(func_name_none.value)
        == "Please provide the name of the function that should be executed by the lambda."
    )

    with pytest.raises(RuntimeError) as empty_name:
        rh.aws_lambda_function(
            paths_to_code=handler_path,
            handler_function_name="",
            runtime="python3.9",
            args_names=["arg1", "arg2"],
            name=name,
        )
    assert (
        str(empty_name.value)
        == "Please provide the name of the function that should be executed by the lambda."
    )


def test_bad_runtime_to_factory():
    name = "test_wrong_runtime"
    handler_path = [f"{TEST_RESOURCES}/basic_test_handler.py"]

    wrong_runtime_1 = rh.aws_lambda_function(
        paths_to_code=handler_path,
        handler_function_name="lambda_sum",
        runtime=None,
        args_names=["arg1", "arg2"],
        name=f"{name}_1",
    )

    wrong_runtime_2 = rh.aws_lambda_function(
        paths_to_code=handler_path,
        handler_function_name="lambda_sum",
        args_names=["arg1", "arg2"],
        name=f"{name}_2",
    )

    wrong_runtime_3 = rh.aws_lambda_function(
        paths_to_code=handler_path,
        handler_function_name="lambda_sum",
        runtime="python3.91",
        args_names=["arg1", "arg2"],
        name=f"{name}_3",
    )

    assert wrong_runtime_1.runtime == "python3.9"
    assert wrong_runtime_2.runtime == "python3.9"
    assert wrong_runtime_3.runtime == "python3.9"
    LAMBDAS_NAMES.add(f"{name}_1")
    LAMBDAS_NAMES.add(f"{name}_2")
    LAMBDAS_NAMES.add(f"{name}_3")


def test_bad_args_names_to_factory(caplog):
    name = "test_lambda_create_and_run"
    handler_path = [f"{TEST_RESOURCES}/basic_test_handler.py"]
    caplog.set_level(logging.ERROR)
    try:
        rh.aws_lambda_function(
            paths_to_code=handler_path,
            handler_function_name="lambda_sum",
            runtime="python3.9",
            args_names=None,
            name=name,
        )
    except RuntimeError:
        assert (
            "Please provide the names of the arguments provided to handler function, in the order they are"
            + " passed to the lambda function."
            in caplog.text
        )

    try:
        rh.aws_lambda_function(
            paths_to_code=handler_path,
            handler_function_name="lambda_sum",
            runtime="python3.9",
            name=name,
        )
    except RuntimeError:
        assert (
            "Please provide the names of the arguments provided to handler function, in the order they are"
            + " passed to the lambda function."
            in caplog.text
        )


def test_func_no_args():
    handler_path = [f"{TEST_RESOURCES}/basic_test_handler.py"]
    name = "test_lambda_no_args"
    my_lambda = rh.aws_lambda_function(
        paths_to_code=handler_path,
        handler_function_name="lambda_no_args",
        runtime="python3.9",
        args_names=[],
        name=name,
    )
    assert my_lambda() == '"no args lambda"'
    LAMBDAS_NAMES.add(name)


def test_create_and_run_generate_name():
    handler_path = [f"{TEST_RESOURCES}/basic_test_handler.py"]
    my_lambda = rh.aws_lambda_function(
        paths_to_code=handler_path,
        handler_function_name="lambda_sum",
        runtime="python3.9",
        args_names=["arg1", "arg2"],
    )
    res = my_lambda(3, 4)
    assert res == "7"
    my_lambda.save()
    reload_func = rh.aws_lambda_function(name="lambda_sum")
    res2 = reload_func(12, 7)
    assert res2 == "19"
    LAMBDAS_NAMES.add(my_lambda.name)


def test_create_and_run_layers_dict():
    handler_path = [f"{TEST_RESOURCES}/basic_handler_layer.py"]
    name = "test_lambda_numpy"
    my_lambda = rh.aws_lambda_function(
        paths_to_code=handler_path,
        handler_function_name="arr_handler",
        runtime="python3.9",
        args_names=["arr1", "arr2"],
        name=name,
        env={"reqs": ["numpy", "pandas"], "env_vars": None},
    )
    my_lambda.save()
    res = my_lambda([1, 2, 3], [1, 2, 3])
    assert res == "12"
    LAMBDAS_NAMES.add(my_lambda.name)


def test_reload_func_with_libs():
    # tests that after the libs are installed, they are not being re-installed.
    my_reloaded_lambda = rh.aws_lambda_function(name="test_lambda_numpy")
    res = my_reloaded_lambda([1, 2, 3], [12, 5, 9])
    assert res == "32"


def test_create_and_run_layers_env():
    handler_path = [f"{TEST_RESOURCES}/basic_handler_layer.py"]
    name = "test_lambda_numpy_env"
    my_env = rh.env(reqs=["numpy"])
    my_lambda = rh.aws_lambda_function(
        paths_to_code=handler_path,
        handler_function_name="arr_handler",
        runtime="python3.9",
        args_names=["arr1", "arr2"],
        name=name,
        env=my_env,
    )
    res = my_lambda([1, 2, 3], [2, 5, 6])
    assert res == "19"
    LAMBDAS_NAMES.add(my_lambda.name)


def test_create_and_run_layers_list():
    handler_path = [f"{TEST_RESOURCES}/basic_handler_layer.py"]
    name = "test_lambda_numpy_list"
    my_lambda = rh.aws_lambda_function(
        paths_to_code=handler_path,
        handler_function_name="arr_handler",
        runtime="python3.9",
        args_names=["arr1", "arr2"],
        name=name,
        env=["numpy"],
    )
    res = my_lambda([1, 2, 3], [4, 7, 9])
    assert res == "26"
    LAMBDAS_NAMES.add(my_lambda.name)


def test_layers_increase_timeout_and_memory():
    handler_path = [f"{TEST_RESOURCES}/basic_handler_layer.py"]
    name = "test_lambda_numpy_increase_params"
    my_lambda = rh.aws_lambda_function(
        paths_to_code=handler_path,
        handler_function_name="arr_handler",
        runtime="python3.9",
        args_names=["arr1", "arr2"],
        name=name,
        env=["numpy"],
        timeout=3,
        memory_size=128,
    )
    res = my_lambda([1, 2, 3], [4, 7, 9])
    assert res == "26"
    lambda_config = LAMBDA_CLIENT.get_function(FunctionName=my_lambda.name)
    assert lambda_config["Configuration"]["Timeout"] == 600
    assert lambda_config["Configuration"]["MemorySize"] == 1024
    assert lambda_config["Configuration"]["EphemeralStorage"]["Size"] == 3072
    assert lambda_config["Configuration"]["FunctionName"] == my_lambda.name
    LAMBDAS_NAMES.add(my_lambda.name)


@pytest.mark.skip(
    "Not sure it is necessary now we are installing libs during runtime. "
)
def test_different_runtimes_and_layers():
    handler_path = [f"{TEST_RESOURCES}/basic_handler_layer.py"]
    name = "test_lambda_numpy"
    my_lambda_37 = rh.aws_lambda_function(
        paths_to_code=handler_path,
        handler_function_name="arr_handler",
        runtime="python3.7",
        args_names=["arr1", "arr2"],
        name=name + "_37",
        env={"reqs": ["numpy"], "env_vars": None},
    )
    res37 = my_lambda_37([1, 2, 3], [2, 5, 6])
    assert res37 == "19"
    LAMBDAS_NAMES.add(my_lambda_37.name)

    my_lambda_38 = rh.aws_lambda_function(
        paths_to_code=handler_path,
        handler_function_name="lambda_handler",
        runtime="python3.8",
        args_names=["arr1", "arr2"],
        name=name + "_38",
        env={"reqs": ["numpy"], "env_vars": None},
    )
    res38 = my_lambda_38([1, 2, 3], [12, 5, 9])
    assert res38 == "32"
    LAMBDAS_NAMES.add(my_lambda_38.name)

    my_lambda_310 = rh.aws_lambda_function(
        paths_to_code=handler_path,
        handler_function_name="lambda_handler",
        runtime="python3.10",
        args_names=["arr1", "arr2"],
        name=name + "_310",
        env={"reqs": ["numpy"], "env_vars": None},
    )
    res310 = my_lambda_310([-2, 5, 1], [12, 5, 9])
    assert res310 == "30"
    LAMBDAS_NAMES.add(my_lambda_310.name)

    my_lambda_311 = rh.aws_lambda_function(
        paths_to_code=handler_path,
        handler_function_name="lambda_handler",
        runtime="python3.11",
        args_names=["arr1", "arr2"],
        name=name + "_311",
        env={"reqs": ["numpy"], "env_vars": None},
    )
    res311 = my_lambda_311([-2, 5, 1], [8, 7, 6])
    assert res311 == "25"
    LAMBDAS_NAMES.add(my_lambda_311.name)


def test_create_and_run_layers_txt():
    handler_path = [f"{TEST_RESOURCES}/basic_handler_layer.py"]
    name = "test_lambda_numpy_txt"
    my_lambda = rh.aws_lambda_function(
        paths_to_code=handler_path,
        handler_function_name="arr_handler",
        runtime="python3.9",
        args_names=["arr1", "arr2"],
        name=name,
        env=f"{os.getcwd()}/test_helpers/lambda_tests/requirements.txt",
    )
    res = my_lambda([1, 2, 3], [1, 2, 3])
    assert res == "12"
    LAMBDAS_NAMES.add(my_lambda.name)


def test_update_lambda_one_file():
    handler_path = [f"{TEST_RESOURCES}/basic_test_handler.py"]
    name = "test_lambda_create_and_run"
    my_lambda = rh.aws_lambda_function(
        paths_to_code=handler_path,
        handler_function_name="lambda_sum",
        runtime="python3.9",
        args_names=["arg1", "arg2"],
        name=name,
    )
    res = my_lambda(6, 4)
    assert res == "10"
    reload_func = rh.aws_lambda_function(name=name)
    res2 = reload_func(12, 13)
    assert res2 == "25"
    LAMBDAS_NAMES.add(my_lambda.name)


def test_mult_files_each():
    """The handler function calls function from each file separately.
    For example, there are a.py, b.py and c.py. Each file has the following funcs, respectively: func_a, func_b and
    func_c. So in the test, the main function (handler) will look something like this:
    import a.py, b.py and c.py
    def handler_func:
        func_a()
        func_b()
        func_c()
    """
    prefix = "call_files_separately"
    folder_path = f"{TEST_RESOURCES}/{prefix}"
    handler_paths = os.listdir(folder_path)
    handler_paths = [p for p in handler_paths if ".py" in p]
    handler_paths.sort()
    handler_paths = [f"{folder_path}/{p}" for p in handler_paths]
    name = "test_lambda_multiple_files_s"
    my_lambda_calc_1 = rh.aws_lambda_function(
        paths_to_code=handler_paths,
        handler_function_name="my_calc",
        runtime="python3.9",
        args_names=["arg1", "arg2"],
        name=name,
        env={"reqs": ["numpy"], "env_vars": None},
    )
    res1 = my_lambda_calc_1(2, 3)
    res2 = my_lambda_calc_1(5, 3)
    res3 = my_lambda_calc_1(2, 7)
    res4 = my_lambda_calc_1(10, 5)
    assert res1 == "2.5"
    assert res2 == "3.2"
    assert res3 == "22.5"
    assert res4 == "7.5"
    LAMBDAS_NAMES.add(my_lambda_calc_1.name)


def test_few_python_files_chain():
    """The handler function calls functions from different files in chain.
    For example, there are a.py, b.py and c.py. Each file has the following funcs, respectively: func_a, func_b and
    func_c. So in the test, the main function (handler) will look something like this:
    import c.py
    def handler_func:
        func_c()
    where func_c() calls func_b() which calls func_a().
    """
    prefix = "call_files_chain"
    folder_path = f"{TEST_RESOURCES}/{prefix}"
    handler_paths = os.listdir(folder_path)
    handler_paths = [p for p in handler_paths if ".py" in p]
    handler_paths.sort()
    handler_paths = [f"{folder_path}/{p}" for p in handler_paths]
    name = "test_lambda_multiple_files_c"
    my_lambda_calc_2 = rh.aws_lambda_function(
        paths_to_code=handler_paths,
        handler_function_name="special_calc",
        runtime="python3.9",
        args_names=["arg1", "arg2"],
        name=name,
    )
    res1 = my_lambda_calc_2(2, 3)
    res2 = my_lambda_calc_2(5, 3)
    res3 = my_lambda_calc_2(2, 7)
    res4 = my_lambda_calc_2(10, 5)
    assert res1 == "16"
    assert res2 == "17"
    assert res3 == "20"
    assert res4 == "20"
    LAMBDAS_NAMES.add(my_lambda_calc_2.name)


def test_args():
    basic_func = rh.aws_lambda_function(name="test_lambda_create_and_run")
    res1 = basic_func(2, 3)
    res2 = basic_func(5, arg2=3)
    res3 = basic_func(arg1=2, arg2=7)
    assert res1 == "5"
    assert res2 == "8"
    assert res3 == "9"


def test_map_starmap():
    basic_func = rh.aws_lambda_function(name="test_lambda_create_and_run")
    res_map1 = basic_func.map([1, 2, 3], [4, 5, 6])
    res_map2 = basic_func.map([6, 2, 3], [15, 52, 61])
    res_map3 = basic_func.starmap([(1, 2), (3, 4), (5, 6)])
    res_map4 = basic_func.starmap([(12, 5), (44, 32), (8, 3)])
    assert res_map1 == ["5", "7", "9"]
    assert res_map2 == ["21", "54", "64"]
    assert res_map3 == ["3", "7", "11"]
    assert res_map4 == ["17", "76", "11"]


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
    config_lambda = rh.AWSLambdaFunction.from_config(config)
    res1 = config_lambda(1, 2)
    res2 = config_lambda(8, 12)
    res3 = config_lambda(14, 17)

    assert res1 == "3"
    assert res2 == "20"
    assert res3 == "31"
    LAMBDAS_NAMES.add(config_lambda.name)


def test_share_lambda():
    basic_func = rh.aws_lambda_function(name="test_lambda_create_and_run")
    users = ["josh@run.house"]
    added_users, new_users = basic_func.share(
        users=users, notify_users=True, access_type="write"
    )
    assert added_users == {}
    assert new_users == {}


def test_remove_resources():
    for lambda_name in LAMBDAS_NAMES:
        policy_name = f"{lambda_name}_Policy"
        role_name = f"{lambda_name}_Role"
        del_policy = IAM_CLIENT.delete_role_policy(
            RoleName=role_name, PolicyName=policy_name
        )
        del_role = IAM_CLIENT.delete_role(RoleName=role_name)
        del_lambda = LAMBDA_CLIENT.delete_function(FunctionName=lambda_name)
        assert del_policy is not None
        assert del_role is not None
        assert del_lambda is not None


if __name__ == "__main__":
    unittest.main()
