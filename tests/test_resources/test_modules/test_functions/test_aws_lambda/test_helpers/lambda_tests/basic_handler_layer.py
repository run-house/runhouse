import numpy as np


def lambda_handler(arr1, arr2):
    arr3 = np.add(arr1, arr2)
    arr3.tolist()
    ret_val = int(arr3[0] + arr3[1] + arr3[2])
    return ret_val
