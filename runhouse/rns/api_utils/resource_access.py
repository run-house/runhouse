from enum import Enum


class ResourceAccess(str, Enum):
    WRITE = "write"
    READ = "read"
    PROXY = "proxy"
