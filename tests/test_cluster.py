import unittest
from pathlib import Path
import pytest

from runhouse.rns import Cluster


def test_setup(tmp_path):
    c = Cluster(name="my_test_cluster", clusters_dir=tmp_path)
    self.assertIsNotNone(c.address)  # add assertion here

def test_teardown_and_delete():
    c = Cluster(name="my_test_cluster", clusters_dir=tmp_path)
    c.teardown()
    self.assertRaises(c.get_cluster_address(create=False),
                      RuntimeError)
    self.assertFalse(Path(tempdir, "my_test_cluster").exists())

def test_teardown_and_delete():
    c = Cluster(name="my_test_cluster", clusters_dir=tempdir)
    c.teardown_and_delete()
    self.assertRaises(c.get_cluster_address(create=False),
                      RuntimeError)

def test_setup_existing_from_name():
    c = Cluster(name="my_test_cluster")
    self.assertIsNotNone(c.address)  # add assertion here

if __name__ == '__main__':
    unittest.main()
