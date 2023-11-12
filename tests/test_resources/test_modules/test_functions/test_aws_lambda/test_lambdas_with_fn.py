def summer(a, b):
    return a + b


def test_create_func():
    import runhouse as rh

    aws_func = rh.aws_lambda_function(fn=summer, name="summer_lambdas").save()
    res = aws_func(1, 3)
    assert int(res) == 4
