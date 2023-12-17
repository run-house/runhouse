import inspect
import logging
import warnings
from pathlib import Path
from typing import Callable, Optional

from runhouse.resources.envs import Env
from runhouse.resources.functions.aws_lambda import LambdaFunction
from runhouse.resources.functions.function import Function

logger = logging.getLogger(__name__)

CRED_PATH = f"{Path.home()}/.aws/credentials"
SUPPORTED_RUNTIMES = [
    "python3.7",
    "python3.8",
    "python3.9",
    "python3.10",
    "python3.11",
]
DEFAULT_PY_VERSION = "python3.9"
LOG_GROUP_PREFIX = "/aws/lambda/"


def aws_lambda_fn(
    fn: Optional[Callable] = None,
    paths_to_code: Optional[list[str]] = None,
    handler_function_name: Optional[str] = None,
    runtime: Optional[str] = None,
    args_names: Optional[list[str]] = None,
    name: Optional[str] = None,
    env: Optional[dict or list[str] or Env] = None,
    timeout: Optional[int] = None,
    memory_size: Optional[int] = None,
    tmp_size: Optional[int] = None,
    retention_time: Optional[int] = None,
    dryrun: bool = False,
):
    """Builds an instance of :class:`LambdaFunction`.

    Args:
        fn (Optional[Callable]): The Lambda function to be executed.
        paths_to_code: (Optional[list[str]]): List of the FULL paths to the python code file(s) that should be sent to
            AWS Lambda. First path in the list should be the path to the handler file which contains the main
            (handler) function. If ``fn`` is provided, this argument is ignored.
        handler_function_name: (Optional[str]): The name of the function in the handler file that will be executed
            by the Lambda. If ``fn`` is provided, this argument is ignored.
        runtime: (Optional[str]): The coding language of the function. Should be one of the following:
            python3.7, python3.8, python3.9, python3.10, python3.11. (Default: ``python3.9``)
        args_names: (Optional[list[str]]): List of the function's accepted parameters, which will be passed to the
            Lambda Function. If your function doesn't accept arguments, please provide an empty list.
            If ``fn`` is provided, this argument is ignored.
        name (Optional[str]): Name of the Lambda Function to create or retrieve.
            This can be either from a local config or from the RNS.
        env (Optional[Dict or List[str] or Env]): Specifies the requirements that will be installed, and the environment
            vars that should be attached to the Lambda. Accepts three possible types:\n
            1. A dict which should contain the following keys:\n
               a. reqs: a list of the python libraries, to be installed by the Lambda, or just a ``requirements.txt``
                  string.\n
               b. env_vars: dictionary containing the env_vars that will be a part of the lambda configuration.\n
            2. A list of strings, containing all the required python packeages.\n
            3. An instance of Runhouse Env class.\n
            By default, ``runhouse`` package will be installed, and env_vars will include ``{HOME: /tmp/home}``.
        timeout: Optional[int]: The maximum amount of time (in secods) during which the Lambda will run in AWS
            without timing-out. (Default: ``900``, Min: ``3``, Max: ``900``)
        memory_size: Optional[int], The amount of memeory (in MB) to be allocated to the Lambda.
             (Default: ``1024``, Min: ``128``, Max: ``10240``)
        tmp_size: Optional[int], This size of the /tmp folder in the aws lambda file system.
             (Default: ``3072``, Min: ``512``, Max: ``10240``).
        retention_time: Optional[int] The time (in days) the Lambda execution logs will be saved in AWS
            cloudwatch. After that, they will be deleted. (Default: ``30`` days)
        dryrun (bool): Whether to create the Function if it doesn't exist, or load the Function object as a dryrun.
            (Default: ``False``).

    Returns:
        LambdaFunction: The resulting AWS Lambda Function object.

        .. note::
            When creating a Lambda function for the first time (not reloading it), the following arguments are
            mandatory: ``paths_to_code`` and ``handler_function_name`` OR a callable function.

    Examples:
        >>> import runhouse as rh

        >>> # handler_file.py
        >>> def summer(a, b):
        >>>    return a + b

        >>> # your 'main' python file, where you are using runhouse
        >>> summer_lambda = rh.aws_lambda_fn(
        >>>                     paths_to_code=['/full/path/to/handler_file.py'],
        >>>                     handler_function_name = 'summer',
        >>>                     runtime = 'python3.9',
        >>>                     name="my_func").save()

        >>> # invoking the function
        >>> summer_res = summer_lambda(5, 8)  # returns "13". (It returns str type because of AWS API)

        >>> # Load function from above
        >>> reloaded_function = rh.aws_lambda_fn(name="my_func")
        >>> reloaded_function_res = reloaded_function(3, 4)  # returns "7".

        >>> # Pass in a callable function  when creating a Lambda
        >>> def multiply(a, b):
        >>>     return a * b
        >>> multiply_lambda = rh.aws_lambda_fn(fn=multiply, name="lambdas_mult_func")
        >>> mult_res = multiply_lambda(4, 5)  # returns "20".

    """
    # TODO: [SB] in the next phase, maybe add the option to create func from git.
    if name and not any(
        [paths_to_code, handler_function_name, runtime, fn, args_names]
    ):
        # Try reloading existing function
        return LambdaFunction.from_name(name=name)

    if not fn or not isinstance(fn, Callable):
        raise RuntimeError(
            "Please provide a callable function OR use from_handler_file method"
            + "in order to create a Lambda function."
        )
    # Env setup.
    env = LambdaFunction.validate_and_create_env(env)

    # extract function pointers, path to code and arg names from callable function.
    handler_function_name = fn.__name__
    fn_pointers = Function._extract_pointers(fn, reqs=[] if env is None else env.reqs)
    paths_to_code = LambdaFunction._paths_to_code_from_fn_pointers(fn_pointers)
    args_names = [param.name for param in inspect.signature(fn).parameters.values()]
    if name is None:
        name = fn.__name__

    # ------- arguments validation -------
    if isinstance(env, str) and "requirements.txt" in env:
        paths_to_code.append(Path(env).absolute())
    if args_names is None:
        # extracting the arguments names of the handler function.
        args_names = LambdaFunction.extract_args_from_file(
            paths_to_code, handler_function_name
        )
    if name is None:
        name = handler_function_name.replace(".", "_")

    # TODO: extract to a seperate method

    if runtime is None or runtime not in SUPPORTED_RUNTIMES:
        warnings.warn(
            f"{runtime} is not a supported by AWS Lambda. Setting runtime to python3.9."
        )
        runtime = DEFAULT_PY_VERSION

    if timeout is None:
        warnings.warn("Timeout set to 15 min.")
        timeout = LambdaFunction.DEFAULT_TIMEOUT
    else:
        if (env.reqs is not None or len(env.reqs) > 0) and timeout < 600:
            warnings.warn(
                "Increasing the timeout to 600 sec, in order to enable the packages setup."
            )
            timeout = 600
        if timeout > 900:
            timeout = 900
            warnings.warn("Timeout can not be more then 900 sec, setting to 900 sec.")
        if timeout < 3:
            timeout = 3
            warnings.warn("Timeout can not be less then 3 sec, setting to 3 sec.")
    if memory_size is None:
        warnings.warn("Memory size set to 1024 MB.")
        memory_size = LambdaFunction.DEFAULT_MEMORY_SIZE
    else:
        if (
            env.reqs is not None or len(env.reqs) > 0
        ) and memory_size < LambdaFunction.DEFAULT_MEMORY_SIZE:
            warnings.warn(
                "Increasing the memory size to 1G, in order to enable the packages setup."
            )
            memory_size = LambdaFunction.DEFAULT_MEMORY_SIZE
        if memory_size < 128:
            memory_size = 128
            warnings.warn("Memory size can not be less then 128 MB, setting to 128 MB.")
        if memory_size > 10240:
            memory_size = 10240
            warnings.warn(
                "Memory size can not be more then 10240 MB, setting to 10240 MB."
            )
    if tmp_size is None or tmp_size < LambdaFunction.DEFAULT_TMP_SIZE:
        warnings.warn(
            "Setting /tmp size to 3GB, in order to enable the packages setup."
        )
        tmp_size = LambdaFunction.DEFAULT_TMP_SIZE
    elif tmp_size > 10240:
        tmp_size = 10240
        warnings.warn("/tmp size can not be more then 10240 MB, setting to 10240 MB.")
    if retention_time is None:
        retention_time = LambdaFunction.DEFAULT_RETENTION

    new_function = LambdaFunction(
        fn_pointers=fn_pointers,
        paths_to_code=paths_to_code,
        handler_function_name=handler_function_name,
        runtime=runtime,
        args_names=args_names,
        name=name,
        env=env,
        dryrun=dryrun,
        timeout=timeout,
        memory_size=memory_size,
        tmp_size=tmp_size,
        retention_time=retention_time,
    )

    if dryrun:
        return new_function

    return new_function.deploy()
