from runhouse.rns import RNSClient

class RhBlob:

    def __init__(self, uri: str, data_url = None: str):
        self.uri = uri
        self.data_url = data_url
        self.rns = RNSClient()
        self.cached_data = None

    def fetch(self, deserializer=None):
        self.data_url = RNSClient.get(self.uri)
        # TODO add support for deserialization
        self.cached_data = self.fetch_from_data_url(self.data_url)
        return self.cached_data

    def set(self, new_data, serializer=None, overwrite=False):
        self.cached_data = new_data
        # TODO add support for serialization
        if overwrite:
            self.data_url = write_to_new_url(self.cached_data)
        else:
            write_to_url(self.data_url, self.cached_data)

    def history(self, entries=10):
        # TODO return the history of this URI, including each new url and which runs have overwritten it.

