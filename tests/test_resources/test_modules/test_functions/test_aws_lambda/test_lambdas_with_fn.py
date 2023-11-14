import os
from pathlib import Path


def summer(a, b):
    return a + b


def multiplier(a: int, b: int):
    return a * b


def test_create_func():
    import runhouse as rh

    aws_func = rh.aws_lambda_function(fn=summer, name="summer_lambdas").save()
    res = aws_func(1, 3)
    assert int(res) == 4
    try:
        os.remove(
            str(Path(aws_func.handler_path).parent / f"rh_handler_{aws_func.name}.py")
        )
    except FileNotFoundError:
        assert True is True
    try:
        os.remove(
            str(Path(aws_func.handler_path).parent / f"{aws_func.name}_code_files.zip")
        )
    except FileNotFoundError:
        assert True is True


def test_from_runhouse_func():
    import runhouse as rh

    my_rh_lambda = rh.function(multiplier).to(system="AWS_LAMBDA")
    res = my_rh_lambda(3, 5)
    assert res == "15"
    try:
        os.remove(
            str(
                Path(my_rh_lambda.handler_path).parent
                / f"rh_handler_{my_rh_lambda.name}.py"
            )
        )
    except FileNotFoundError:
        assert True is True
    try:
        os.remove(
            str(
                Path(my_rh_lambda.handler_path).parent
                / f"{my_rh_lambda.name}_code_files.zip"
            )
        )
    except FileNotFoundError:
        assert True is True
