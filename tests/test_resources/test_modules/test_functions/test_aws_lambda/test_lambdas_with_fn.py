import runhouse as rh
from runhouse.globals import rns_client


def summer(a, b):
    return a + b


def multiplier(a: int, b: int):
    return a * b


def test_create_func():
    aws_func = rh.aws_lambda_fn(fn=summer, name="summer_lambdas")
    aws_func.save()
    res = aws_func(1, 3)
    assert res == 4
    aws_func.teardown()
    rns_client.delete_configs(aws_func)


def test_from_runhouse_func():
    my_rh_lambda = rh.function(multiplier).to(system="Lambda_Function")
    res = my_rh_lambda(3, 5)
    assert res == 15
    my_rh_lambda.teardown()


def test_share_lambda():
    user = rh.configs.username
    from tests.utils import friend_account

    with friend_account():
        lambda_func = rh.aws_lambda_fn(fn=summer, name="summer_to_share")
        lambda_func.save()
        lambda_func.share(users=[user], notify_users=False, access_level="write")
        shared_func = lambda_func.rns_address
    reloaded_func = rh.aws_lambda_fn(name=shared_func)
    assert reloaded_func(1, 3) == 4
    rns_client.delete_configs(lambda_func)
    lambda_func.teardown()
