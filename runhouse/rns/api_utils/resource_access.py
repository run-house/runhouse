from enum import Enum


class ResourceAccess(str, Enum):
    write = "write"
    read = "read"
    proxy = "proxy"
