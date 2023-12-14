import sum_file


def sub_lambda(arg1: int, arg2: int):
    res = abs(arg1 - arg2)
    res1 = sum_file.lambda_sum(15, res)
    return res1
