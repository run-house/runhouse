import runhouse as rh


def summer(a, b):
    return a + b


def multiplier(a: int, b: int):
    return a * b


def test_create_func():
    aws_func = rh.aws_lambda_function(fn=summer, name="summer_lambdas")
    aws_func.save()
    res = aws_func(1, 3)
    assert int(res) == 4
    aws_func.delete()


def test_from_runhouse_func():
    my_rh_lambda = rh.function(multiplier).to(system="AWS_LAMBDA")
    res = my_rh_lambda(3, 5)
    assert res == "15"
    my_rh_lambda.delete()


def test_share_lambda(test_account):
    user = rh.configs.get("username")
    with test_account:
        lambda_func = rh.aws_lambda_function(fn=summer, name="summer_to_share")
        lambda_func.save()
        lambda_func.share(users=[user], notify_users=True, access_type="write")
        shared_func = lambda_func.rns_address
    reloaded_func = rh.aws_lambda_function(name=shared_func)
    assert reloaded_func(1, 3) == "4"
    lambda_func.delete()
