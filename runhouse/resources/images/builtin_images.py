from runhouse.resources.images.image import Image


def dask():
    return Image().install_packages(["dask[distributed,dataframe]", "dask-ml"])


def pytorch():
    return Image().install_packages(["torch"])


def ray():
    return Image().install_packages(["ray[tune,data,train]"])
