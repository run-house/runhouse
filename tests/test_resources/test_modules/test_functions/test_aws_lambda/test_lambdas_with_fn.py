import runhouse as rh


def summer(a, b):
    return a + b


def multiplier(a: int, b: int):
    return a * b


def test_create_func():
    aws_func = rh.aws_lambda_function(fn=summer, name="summer_lambdas").save()
    res = aws_func(1, 3)
    assert int(res) == 4


def test_from_runhouse_func():
    my_rh_lambda = rh.function(multiplier).to(system="AWS_LAMBDA")
    res = my_rh_lambda(3, 5)
    assert res == "15"
