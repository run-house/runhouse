from typing import AsyncIterable, Awaitable, Generator

from runhouse.resources.module import Module
from runhouse.servers.obj_store import RunhouseStopIteration


class FutureModule(Module, Awaitable):
    """A module that represents a future result, but capable of representing a result on a remote system."""

    def __init__(self, future, **kwargs):
        super().__init__(**kwargs)
        self._future = future

    def result(self):
        return self._future.result()

    def remote_await(self):
        return self._future

    def __await__(self):
        # TODO talk about module dunder stuff through this file
        # __await__ needs to return an awaitable, so we wrap it in a coroutine
        async def __await():
            return self.remote_await(run_name=self.name)

        return __await().__await__()

    def set_result(self, __result):
        return self._future.set_result(__result)

    def set_exception(self, __exception):
        return self._future.set_exception(__exception)

    def done(self):
        return self._future.done()

    def cancelled(self):
        return self._future.cancelled()

    def add_done_callback(self, __fn, *, context=None):
        return self._future.add_done_callback(__fn, context=context)

    def remove_done_callback(self, __fn):
        return self._future.remove_done_callback(__fn)

    def cancel(self, msg=None):
        return self._future.cancel(msg=msg)

    def exception(self):
        return self._future.exception()

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_future"] = None
        return state


class GeneratorModule(Module, Generator):
    """A module that represents a future result, but capable of representing a result on a remote system."""

    def __init__(self, future, **kwargs):
        super().__init__(**kwargs)
        self._future = future

    def __iter__(self):
        # Note that we can't return self._future.__iter__() here because that would be a generator,
        # which we can't pickle. Instead, we use __next__-style generation.
        return self

    def remote_next(self):
        try:
            return self._future.__next__()
        except StopIteration:
            raise RunhouseStopIteration()

    def __next__(self):
        return self.remote_next()

    def send(self, __value):
        return self._future.send(__value)

    def throw(self, __typ, __val=None, __tb=None):
        return self._future.throw(__typ, __val, __tb)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_future"] = None
        return state


class AsyncGeneratorModule(Module, AsyncIterable):
    """A module that represents a future result, but capable of representing a result on a remote system."""

    def __init__(self, future, **kwargs):
        super().__init__(**kwargs)
        self._future = future

    def __aiter__(self):
        return self

    def remote_anext(self):
        return self._future.__anext__()

    async def __anext__(self):
        return self.remote_anext(run_name=self.name)

    async def remote_await(self):
        return await self._future

    def __await__(self):
        return self.remote_await()

    def send(self, __value):
        return self._future.send(__value)

    def throw(self, __typ, __val=None, __tb=None):
        return self._future.throw(__typ, __val, __tb)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_future"] = None
        return state
