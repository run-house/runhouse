import inspect
from typing import Any, Optional, Tuple, Union

from runhouse.resources.envs import _get_env_from, Env
from runhouse.resources.module import Module, MODULE_ATTRS


class Asgi(Module):
    def __init__(self, app_pointers: Optional[Tuple] = None, **kwargs):
        """
        Runhouse Asgi Module. Allows you to deploy an ASGI app to a cluster.

        .. note::
                To create an Asgi, please use the factory method :func:`asgi`.
        """
        super().__init__(**kwargs)
        self.app_pointers = app_pointers
        self._app = None

    async def asgi_call(self, scope, receive, send) -> None:
        if not self._app:
            self._app = self._get_obj_from_pointers(*self.app_pointers)
        await self._app(scope, receive, send)

    def __call__(self, *args, **kwargs):
        return self.asgi_call(*args, **kwargs)

    def _route_names_and_endpoints(self):
        if not self._app:
            self.local._app = Module._get_obj_from_pointers(*self.local.app_pointers)

        return {route.name: route.endpoint for route in self.local._app.routes[4:]}

    def signature(self, rich=False):
        sig = super().signature(rich=rich)

        route_attrs = {
            name: self.method_signature(endpoint) if rich else None
            for name, endpoint in self._route_names_and_endpoints().items()
            if not name[0] == "_"
            and name not in MODULE_ATTRS
            and name not in dir(Module)
            and callable(endpoint)
            and not (
                "local" in endpoint.parameters and endpoint.parameters["local"].default
            )
        }
        sig.update(route_attrs)
        return sig

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_app"] = None
        return state

    def _signature_extensions(self, item):
        if item == "_app":
            # Object hasn't been remotely initialized yet
            return None

        # If the item is a route name, return the endpoint. This allows us to call the route name directly
        # as a method, which is nice for debugging.
        return self._route_names_and_endpoints().get(item)


def asgi(
    app: Any,
    env: Optional[Union[str, "Env"]] = None,
    name: Optional[str] = None,
    load_from_den: bool = True,
    dryrun: bool = False,
    **kwargs
) -> Asgi:
    """
    A factory method for creating ASGI modules, such as a FastAPI or Starlette app.

    Args:
        app (Any): The ASGI app to deploy.
        **kwargs: Keyword arguments for the :class:`Asgi` constructor.

    Returns:
        Asgi: The resulting Asgi object.

    Example:
        >>> from fastapi import FastAPI
        >>> import runhouse as rh
        >>> app = FastAPI()
        >>> @app.get("/summer/{a}")
        >>> def summer(a: int, b: int = 2):
        >>>     return a + b
        >>>
        >>> fast_api_module = rh.asgi(app).to(rh.here, name="fast_api_module")
        >>> fast_api_module.summer(1, 2)
        >>> # output: 3
    """
    if name and not any([app, kwargs, env]):
        # Try reloading existing module
        return Asgi.from_name(name, load_from_den=load_from_den, dryrun=dryrun)

    if not isinstance(env, Env):
        env = _get_env_from(env) or Env()
        env.working_dir = env.working_dir or "./"

    callers_global_vars = inspect.currentframe().f_back.f_globals.items()
    app_name = [
        var_name for var_name, var_val in callers_global_vars if var_val is app
    ][0]

    app_file, app_module, _ = Module._extract_pointers(
        app.routes[4].endpoint, reqs=env.reqs
    )

    return Asgi(
        app_pointers=(app_file, app_module, app_name),
        env=env,
        name=name,
        dryrun=dryrun,
        **kwargs
    )
