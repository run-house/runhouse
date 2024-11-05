from abc import abstractmethod

from runhouse.resources.module import Module, MODULE_ATTRS, MODULE_METHODS
from runhouse.utils import client_call_wrapper


class Supervisor(Module):
    def __init__(self, name, **kwargs):
        super().__init__(name=name, **kwargs)

    @abstractmethod
    def forward(self, item, *args, **kwargs):
        pass

    def __getattribute__(self, item):
        """Override to allow for remote execution if system is a remote cluster. If not, the subclass's own
        __getattr__ will be called."""
        if (
            item in MODULE_METHODS
            or item in MODULE_ATTRS
            or not hasattr(self, "_client")
        ):
            return super().__getattribute__(item)

        try:
            name = super().__getattribute__("_name")
        except AttributeError:
            return super().__getattribute__(item)

        if item not in self.signature(rich=False) or not name:
            return super().__getattribute__(item)

        # Do this after the signature and name check because it's potentially expensive
        client = super().__getattribute__("_client")()

        if not client:
            return super().__getattribute__(item)

        system = super().__getattribute__("_system")

        is_coroutine_function = (
            self.signature(rich=True)[item]["async"]
            and not self.signature(rich=True)[item]["gen"]
        )

        class RemoteMethodWrapper:
            """Helper class to allow methods to be called with __call__, remote, or run."""

            def __call__(self, *args, **kwargs):
                # stream_logs and run_name are both supported args here, but we can't include them explicitly because
                # the local code path here will throw an error if they are included and not supported in the
                # method signature.

                # Always take the user overrided behavior if they want it
                if "run_async" in kwargs:
                    run_async = kwargs.pop("run_async")
                else:
                    run_async = is_coroutine_function

                if run_async:
                    return client_call_wrapper(
                        client,
                        system,
                        "acall",
                        name,
                        "forward",
                        run_name=kwargs.pop("run_name", None),
                        stream_logs=kwargs.pop("stream_logs", True),
                        remote=kwargs.pop("remote", False),
                        data={"args": [item] + (args or []), "kwargs": kwargs},
                    )
                else:
                    args = args or []
                    return client_call_wrapper(
                        client,
                        system,
                        "call",
                        name,
                        "forward",
                        run_name=kwargs.pop("run_name", None),
                        stream_logs=kwargs.pop("stream_logs", True),
                        remote=kwargs.pop("remote", False),
                        data={"args": [item] + list(args), "kwargs": kwargs},
                    )

            def remote(self, *args, stream_logs=True, run_name=None, **kwargs):
                return self.__call__(
                    *args,
                    stream_logs=stream_logs,
                    run_name=run_name,
                    remote=True,
                    **kwargs,
                )

            def run(self, *args, stream_logs=False, run_name=None, **kwargs):
                return self.__call__(
                    *args,
                    stream_logs=stream_logs,
                    run_name=run_name,
                    **kwargs,
                )

        return RemoteMethodWrapper()
