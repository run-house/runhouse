import logging
import os
import shutil
import time
import unittest

import boto3
import pytest
import runhouse as rh

logger = logging.getLogger(__name__)
CUR_WORK_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_RESOURCES = f"{CUR_WORK_DIR}/test_resources/lambda_tests"
LAMBDA_CLIENT = boto3.client("lambda")
IAM_CLIENT = boto3.client("iam")


@pytest.fixture(scope="session", autouse=True)
def download_resources():
    curr_folder = os.getcwd()
    s3_resource = boto3.resource("s3")
    bucket = s3_resource.Bucket("runhouse-lambda-resources")
    remoteDirectoryName = "test_resources/lambda_tests"
    objs = bucket.objects.filter(Prefix=remoteDirectoryName)
    for obj in objs:
        dir_name = "/".join(obj.key.split("/")[:-1])
        if not os.path.exists(f"{curr_folder}/{dir_name}"):
            os.makedirs(f"{curr_folder}/{dir_name}")
        bucket.download_file(obj.key, f"{curr_folder}/{obj.key}")


# def delete_aws_resources():
#     lambda_role, lambda_arn, policy_arn = "", "", ""
#     del_policy = IAM_CLIENT.delete_role(RoleName=lambda_role)
#     del_role = IAM_CLIENT.delete_role(RoleName=lambda_role)
#     del_lambda = LAMBDA_CLIENT.delete_function(FunctionName=lambda_arn)
#     assert del_role is not None
#     assert del_lambda is not None


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

    time.sleep(5)  # letting the lambda be updated in AWS.
    my_lambda.save()
    res = my_lambda(3, 4)
    assert res == "7"
    reload_func = rh.aws_lambda_function(name=name)
    res2 = reload_func(12, 7)
    assert res2 == "19"


def test_load_not_existing_lambda():
    name = "test_lambda_create_and_run1"
    my_lambda = rh.aws_lambda_function(name=name)
    assert my_lambda == "LambdaNotFoundInAWS"


def test_crate_no_arguments():
    res = rh.aws_lambda_function()
    assert res == "NoEnoughArgsProvided"


def test_create_and_run_generate_name():
    handler_path = [f"{TEST_RESOURCES}/basic_test_handler.py"]
    my_lambda = rh.aws_lambda_function(
        paths_to_code=handler_path,
        handler_function_name="lambda_sum",
        runtime="python3.9",
        args_names=["arg1", "arg2"],
    )
    time.sleep(5)  # letting the lambda be updated in AWS.
    res = my_lambda(3, 4)
    assert res == "7"
    my_lambda.save()
    reload_func = rh.aws_lambda_function(name="lambda_sum")
    time.sleep(1)
    res2 = reload_func(12, 7)
    assert res2 == "19"


def test_create_and_run_layers():
    handler_path = [f"{TEST_RESOURCES}/basic_handler_layer.py"]
    name = "test_lambda_numpy"
    my_lambda = rh.aws_lambda_function(
        paths_to_code=handler_path,
        handler_function_name="lambda_handler",
        runtime="python3.9",
        args_names=["arr1", "arr2"],
        name=name,
        env=["numpy", "pandas"],
    )
    time.sleep(4)  # letting the lambda be updated in AWS.
    res = my_lambda([1, 2, 3], [1, 2, 3])
    assert res == "12"


def test_different_runtimes_and_layers():
    handler_path = [f"{TEST_RESOURCES}/basic_handler_layer.py"]
    name = "test_lambda_numpy"
    my_lambda_37 = rh.aws_lambda_function(
        paths_to_code=handler_path,
        handler_function_name="lambda_handler",
        runtime="python3.7",
        args_names=["arr1", "arr2"],
        name=name + "_37",
        env=["numpy", "pandas"],
    )
    time.sleep(4)  # letting the lambda be updated in AWS.
    res37 = my_lambda_37([1, 2, 3], [2, 5, 6])
    assert res37 == "19"

    my_lambda_38 = rh.aws_lambda_function(
        paths_to_code=handler_path,
        handler_function_name="lambda_handler",
        runtime="python3.8",
        args_names=["arr1", "arr2"],
        name=name + "_38",
        env=["numpy", "pandas"],
    )
    time.sleep(4)  # letting the lambda be updated in AWS.
    res38 = my_lambda_38([1, 2, 3], [12, 5, 9])
    assert res38 == "32"

    my_lambda_310 = rh.aws_lambda_function(
        paths_to_code=handler_path,
        handler_function_name="lambda_handler",
        runtime="python3.10",
        args_names=["arr1", "arr2"],
        name=name + "_310",
        env=["numpy", "pandas"],
    )
    time.sleep(4)  # letting the lambda be updated in AWS.
    res310 = my_lambda_310([-2, 5, 1], [12, 5, 9])
    assert res310 == "30"

    my_lambda_311 = rh.aws_lambda_function(
        paths_to_code=handler_path,
        handler_function_name="lambda_handler",
        runtime="python3.11",
        args_names=["arr1", "arr2"],
        name=name + "_311",
        env=["numpy", "pandas"],
    )
    time.sleep(4)  # letting the lambda be updated in AWS.
    res311 = my_lambda_311([-2, 5, 1], [8, 7, 6])
    assert res311 == "25"


def test_create_and_run_layers_txt():
    handler_path = [f"{TEST_RESOURCES}/basic_handler_layer.py"]
    name = "test_lambda_numpy_txt"
    my_lambda = rh.aws_lambda_function(
        paths_to_code=handler_path,
        handler_function_name="lambda_handler",
        runtime="python3.9",
        args_names=["arr1", "arr2"],
        name=name,
        env=f"{os.getcwd()}/test_resources/lambda_tests/requirements.txt",
    )
    time.sleep(5)  # letting the lambda be updated in AWS.
    res = my_lambda([1, 2, 3], [1, 2, 3])
    assert res == "12"


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
    time.sleep(5)  # letting the lambda be updated in AWS.
    res = my_lambda(6, 4)
    assert res == "10"
    reload_func = rh.aws_lambda_function(name=name)
    time.sleep(1)
    res2 = reload_func(12, 13)
    assert res2 == "25"


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
        env=["numpy"],
    )
    time.sleep(4)  # letting the lambda be updated in AWS.
    res1 = my_lambda_calc_1(2, 3)
    res2 = my_lambda_calc_1(5, 3)
    res3 = my_lambda_calc_1(2, 7)
    res4 = my_lambda_calc_1(10, 5)
    assert res1 == "2.5"
    assert res2 == "3.2"
    assert res3 == "22.5"
    assert res4 == "7.5"


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
    time.sleep(4)  # letting the lambda be updated in AWS.
    res1 = my_lambda_calc_2(2, 3)
    res2 = my_lambda_calc_2(5, 3)
    res3 = my_lambda_calc_2(2, 7)
    res4 = my_lambda_calc_2(10, 5)
    assert res1 == "16"
    assert res2 == "17"
    assert res3 == "20"
    assert res4 == "20"


def test_args():
    basic_func = rh.aws_lambda_function(name="test_lambda_create_and_run")
    time.sleep(1)
    res1 = basic_func(2, 3)
    res2 = basic_func(5, arg2=3)
    res3 = basic_func(arg1=2, arg2=7)
    assert res1 == "5"
    assert res2 == "8"
    assert res3 == "9"


def test_map_starmap():
    basic_func = rh.aws_lambda_function(name="test_lambda_create_and_run")
    time.sleep(1)
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
    }
    config_lambda = rh.AWSLambdaFunction.from_config(config)
    time.sleep(4)  # letting the lambda be updated in AWS.
    res1 = config_lambda(1, 2)
    res2 = config_lambda(8, 12)
    res3 = config_lambda(14, 17)

    assert res1 == "3"
    assert res2 == "20"
    assert res3 == "31"


def test_share_lambda():
    basic_func = rh.aws_lambda_function(name="test_lambda_create_and_run")
    time.sleep(1)
    users = ["josh@run.house"]
    added_users, new_users = basic_func.share(
        users=users, notify_users=True, access_type="write"
    )
    assert added_users == {}
    assert new_users == {}


def test_remove_resources():
    curr_folder = os.getcwd()
    remoteDirectoryName = "test_resources"
    shutil.rmtree(f"{curr_folder}/{remoteDirectoryName}")


if __name__ == "__main__":
    download_resources()
    unittest.main()
