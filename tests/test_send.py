import unittest
import runhouse as rh


class TestSend(unittest.TestCase):

    def test_create_send(self):
        def summer(a, b):
            return a + b
        remote_sum = rh.Send(summer)
        res = remote_sum(1, 5)
        self.assertEqual(res, 6)

    def test_remote_send(self):
        # l = lambda x: x + 1
        def summer(a, b):
            return a + b
        re_fn = rh.Send(summer, cluster='default')
        res = re_fn(5, 1)
        self.assertEqual(res, 6)

    def test_remote_send_with_dep(self):
        # import numpy as np
        import torch

        def torch_summer(a, b):
            return torch.Tensor([a, b]).sum()
        re_fn = rh.Send(torch_summer, cluster='default', reqs=['torch'])
        res = re_fn(5, 1)
        self.assertEqual(res, 6)

    def test_remote_send_with_custom_hw(self):
        # import numpy as np
        import torch

        def torch_summer(a, b):
            return torch.Tensor([a, b]).sum()
        re_fn = rh.Send(torch_summer, cluster='default', reqs=['torch'], hardware='rh_8_cpu')
        res = re_fn(5, 1)
        self.assertEqual(res, 6)


if __name__ == '__main__':
    unittest.main()
