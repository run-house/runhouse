import ray


def send(fn):
    ray.init(local_mode=True)
    remote_fn = ray.remote(fn)

    def sent_fn(*args):
        res = remote_fn.remote(*args)
        return ray.get(res)
    return sent_fn
