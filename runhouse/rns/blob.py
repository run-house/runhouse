from runhouse.rns import RNSClient

from datetime import datetime

import fsspec

class RhBlob:

    def __init__(self, uri: str, data_url = None: str, data_config = None):
        self._uri = uri
        self._data_url = data_url
        self._data_config = data_config
        self._rns = RNSClient()
        # TODO this caching is dumb, replace with caching from fsspec
        self._cached_data = None

    @property
    def uri(self)
        return self._uri

    @property
    def data_url(self)
        return self._data_url

    @property
    def data_config(self)
        return self._data_config

    @uri.setter
    def uri(self, value):
        self._uri = value
        self.write_to_rns()
        #TODO add to some "renamed" table?

    @data_url.setter
    def data_url(self, value):
        self._data_url = value
        self.write_to_rns()

    @data_config.setter
    def data_config(self, value):
        self._data_config = value
        self.write_to_rns()

    def write_to_rns(self):
        self._rns.set(self.uri, {'url': self.data_url, 'config': self.data_config})

    def fetch_from_rns(self):
        data_rns_entry = RNSClient.get(self.uri)
        self.data_url = data_rns_entry['url']
        self.data_config = data_rns_entry['config']

    def fetch(self, deserializer = None, return_file_like = False):
        self.fetch_from_rns()
        fss_file = fsspec.open(self.data_url, self.data_config**)
        if return_file_like:
            return fss_file
        with fss_file as f:
            if deserializer is not None:
                self.cached_data = deserializer(f.read())
            else:
                self.cached_data = f.read()
        return self.cached_data

    def save(self, new_data, serializer = None, overwrite = False):
        self.cached_data = new_data
        self.fetch_from_rns()
        if not overwrite:
            time = datetime.today().strftime('%Y-%m-%d_%H:%M:%S')
            self.data_url = self.data_url + time or time 
        fss_file = fsspec.open(self.data_url, mode='wb', self.data_config**)
        with fss_file as f:
            if serializer is not None:
                f.write(serializer(new_data))
            else:
                f.write(new_data)

    def history(self, entries=10):
        # TODO return the history of this URI, including each new url and which runs have overwritten it.

