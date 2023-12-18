import runhouse as rh


def summer(a, b):
    return a + b


def multiplier(a: int, b: int):
    return a * b


def test_create_func():
    aws_func = rh.aws_lambda_fn(fn=summer, name="summer_lambdas")
    aws_func.save()
    res = aws_func(1, 3)
    assert int(res) == 4
    aws_func.delete_from_den()


def test_from_runhouse_func():
    my_rh_lambda = rh.function(multiplier).to(system="Lambda_Function")
    res = my_rh_lambda(3, 5)
    assert res == "15"
    my_rh_lambda.delete()


def test_share_lambda(test_account):
    user = rh.configs.get("username")
    with test_account:
        lambda_func = rh.aws_lambda_fn(fn=summer, name="summer_to_share")
        lambda_func.save()
        lambda_func.share(users=[user], notify_users=True, access_level="write")
        shared_func = lambda_func.rns_address
    reloaded_func = rh.aws_lambda_fn(name=shared_func)
    assert reloaded_func(1, 3) == "4"
    lambda_func.delete_from_den()
