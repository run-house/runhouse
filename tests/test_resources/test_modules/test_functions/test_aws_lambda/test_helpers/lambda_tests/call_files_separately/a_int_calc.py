import div_file
import mult_file
import numpy as np
import sub_file
import sum_file


def my_calc(arg1: int, arg2: int):
    np.array([1, 2])
    res1 = sum_file.lambda_sum(arg1, arg2)
    res2 = abs(sub_file.sub_lambda(arg1, arg2))
    res3 = mult_file.mult_lambda(res1, res2)
    res4 = div_file.div_lambda(res3, arg1)
    return res4
