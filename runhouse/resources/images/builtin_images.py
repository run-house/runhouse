from runhouse.resources.images.image import Image


def dask():
    return Image().pip_install(["dask[distributed,dataframe]", "dask-ml"])


def pytorch():
    return Image().pip_install(["torch"])


def ray():
    return Image().pip_install(["ray[tune,data,train]"])
