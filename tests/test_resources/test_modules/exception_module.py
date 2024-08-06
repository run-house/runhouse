import plotly  # noqa
import runhouse as rh


class ExceptionModule(rh.Module):
    def __init__(self):
        super().__init__()

    def test_fn(self):
        return None
