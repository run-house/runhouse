import unittest
import runhouse as rh


class TestSend(unittest.TestCase):

    def test_create_send(self):
        sum = lambda a, b: a + b
        remote_sum = rh.send(sum)
        res = remote_sum(1, 5)
        self.assertEqual(res, 6)  # add assertion here


if __name__ == '__main__':
    unittest.main()
