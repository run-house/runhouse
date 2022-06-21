import json
import os
import redis

class RNSClient:

    def __init__(self):
        self.redis = redis.Redis()
        try:
            self.redis.ping()
        except redis.exceptions.ConnectionError as e:
            raise ConnectionError(f'Unable to connect to URI service: {e}')

    def get(self, uri: str):
        # Do this properly with hset to avoid using json for no reason
        data = self.redis.get(uri)
        return json.loads(data)

    def set(self, uri: str, data) -> None:
        self.redis.set(uri, json.dumps(data))

    def exists(self, uri: str) -> bool:
        # Do this properly with hset to avoid using json for no reason
        return self.redis.get(uri) is not None
