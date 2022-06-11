import json
import os
from pathlib import Path

from runhouse.utils.utils import current_time

PATH_TO_DB = os.path.join(os.getcwd(), Path(__file__).parent, 'database.json')


class DatabaseAPI:

    def __init__(self, uri):
        self.uri = uri
        # "Connect" to the DB right away - loading in the db contents into memory
        self.redis_db = self.connect_to_db()

    @property
    def db_key(self):
        # TODO in the future account for user / hardware / etc. when building the key?
        # import hashlib
        # hashlib.md5(f'{user}-{name}-{hardware}'.encode('utf-8')).hexdigest()
        return self.uri

    def key_exists_in_db(self) -> bool:
        if self.db_key in self.redis_db:
            return True
        return False

    @staticmethod
    def connect_to_db() -> dict:
        try:
            # TODO swap this out with real redis
            with open(PATH_TO_DB, "r") as f:
                db_data = json.load(f)
                return db_data
        except Exception as e:
            raise ConnectionError(f'Unable to connect to redis: {e}')

    def update_db(self):
        # TODO swap this out with real redis
        with open(PATH_TO_DB, "w") as f:
            json.dump(self.redis_db, f)

    def add_cached_uri_to_db(self, hardware, code):
        self.redis_db[self.db_key] = {'hardware': hardware,
                                      'code': code,
                                      'date_added': current_time()}
        self.update_db()
