from typing import AsyncIterable, Awaitable, AwaitableGenerator, Generator

from runhouse.resources.module import Module


class FutureModule(Module, Awaitable):
    """A module that represents a future result, but capable of representing a result on a remote system."""

    def __init__(self, future, **kwargs):
        super().__init__(**kwargs)
        self.future = future

    def result(self):
        return self.future.result()

    def remote_await(self):
        return self.future.__await__()

    def __await__(self):
        # TODO talk about module dunder stuff through this file
        return self.remote_await()

    def set_result(self, __result):
        return self.future.set_result(__result)

    def set_exception(self, __exception):
        return self.future.set_exception(__exception)

    def done(self):
        return self.future.done()

    def cancelled(self):
        return self.future.cancelled()

    def add_done_callback(self, __fn, *, context=None):
        return self.future.add_done_callback(__fn, context=context)

    def remove_done_callback(self, __fn):
        return self.future.remove_done_callback(__fn)

    def cancel(self, msg=None):
        return self.future.cancel(msg=msg)

    def exception(self):
        return self.future.exception()


class GeneratorModule(Module, Generator):
    """A module that represents a future result, but capable of representing a result on a remote system."""

    def __init__(self, future, **kwargs):
        super().__init__(**kwargs)
        self.future = future

    def __iter__(self):
        # Note that we can't return self.future.__iter__() here because that would be a generator,
        # which we can't pickle. Instead, we use __next__-style generation.
        return self

    def remote_next(self):
        return self.future.__next__()

    def __next__(self):
        return self.remote_next()

    def send(self, __value):
        return self.future.send(__value)

    def throw(self, __typ, __val=None, __tb=None):
        return self.future.throw(__typ, __val, __tb)


class AsyncGeneratorModule(Module, AwaitableGenerator, AsyncIterable):
    """A module that represents a future result, but capable of representing a result on a remote system."""

    def __init__(self, future, **kwargs):
        super().__init__(**kwargs)
        self.future = future

    def __aiter__(self):
        return self

    def remote_anext(self):
        return self.future.__anext__()

    def __anext__(self):
        return self.remote_anext()

    def remote_await(self):
        return self.future.__await__()

    def __await__(self):
        return self.remote_await()

    def send(self, __value):
        return self.future.send(__value)

    def throw(self, __typ, __val=None, __tb=None):
        return self.future.throw(__typ, __val, __tb)
