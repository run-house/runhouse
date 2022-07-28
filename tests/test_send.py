import unittest
import runhouse as rh


class TestSend(unittest.TestCase):

    def test_create_send(self):
        def summer(a, b):
            return a + b
        remote_sum = rh.send(summer)
        res = remote_sum(1, 5)
        self.assertEqual(res, 6)


if __name__ == '__main__':
    unittest.main()
