import json
import os
import redis
from pathlib import Path


class RNSClient:

    def __init__(self, ):
        self.redis = redis.Redis()
        try:
            self.redis.ping()
        except redis.exceptions.ConnectionError as e:
            raise ConnectionError(f'Unable to connect to RNS service: {e}')

    def load_config_from_name(self, name, resource_dir, resource_type):
        # Create the resource config directory, e.g. ~/.rh/clusters/<my_cluster_name>
        name_dir = Path(resource_dir)
        name_dir.mkdir(parents=True, exist_ok=True)
        config_path = name_dir / "config.json"

        # For now, don't hit API at all if config is present.
        # Maybe later check API in case any args are missing or if fresher version is available.
        if config_path.is_file():
            config = json.load(config_path.open('r'))
        else:
            # TODO pull yaml (and save down) and address from real API
            uri = resource_type + ":" + name
            config = self.get(uri)

        # TODO print to logger that config was loaded from x place or uri

        return config

    def save_config_for_name(self, name, config, resource_dir, resource_type):
        name_dir = Path(resource_dir)
        name_dir.mkdir(parents=True, exist_ok=True)
        config_path = name_dir / "config.json"

        json.dump(config, config_path.open('w'))
        uri = resource_type + ":" + name
        self.set(uri, config)

        # TODO print to logger that config was saved to x place
        # print(f"Cluster config saved to {config_path} and Runhouse URI <>")

    def delete_configs(self, name, resource_dir, resource_type):
        if resource_dir.exists():
            resource_dir.rmdir()
        uri = resource_type + ":" + name
        self.delete(uri)

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


def set_working_dir(name):
    os.environ['RH_WORKING_DIR'] = name
