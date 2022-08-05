import json
import os
import redis


class RNSClient:

    def __init__(self, ):
        self.redis = redis.Redis()
        try:
            self.redis.ping()
        except redis.exceptions.ConnectionError as e:
            raise ConnectionError(f'Unable to connect to RNS service: {e}')

    def get(self, uri: str) -> dict:
        # Do this properly with hset to avoid using json for no reason
        return self.redis.hgetall(uri)

    def set(self, uri: str, data: dict) -> None:
        self.redis.hset(uri, mapping=data)

    def exists(self, uri: str) -> bool:
        # Do this properly with hset to avoid using json for no reason
        return self.redis.get(uri) is not None

    def delete(self, uri: str):
        self.redis.delete(uri)
