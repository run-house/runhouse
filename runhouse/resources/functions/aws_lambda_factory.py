from pathlib import Path
from typing import Callable, Dict, List, Optional, Union

from runhouse.resources.envs import Env
from runhouse.resources.functions.aws_lambda import LambdaFunction
from runhouse.resources.functions.function import Function
from runhouse.utils import extract_module_path


CRED_PATH = f"{Path.home()}/.aws/credentials"
DEFAULT_PY_VERSION = "python3.9"
LOG_GROUP_PREFIX = "/aws/lambda/"


def aws_lambda_fn(
    fn: Optional[Callable] = None,
    name: Optional[str] = None,
    env: Optional[Union[Dict, List[str], Env]] = None,
    runtime: Optional[str] = None,
    timeout: Optional[int] = None,
    memory_size: Optional[int] = None,
    tmp_size: Optional[int] = None,
    retention_time: Optional[int] = None,
    load_from_den: bool = True,
    dryrun: bool = False,
):
    """Builds an instance of :class:`LambdaFunction`.

    Args:
        fn (Optional[Callable]): The Lambda function to be executed.
        name (Optional[str]): Name of the Lambda Function to create or retrieve.
            This can be either from a local config or from the RNS.
        env (Optional[Dict or List[str] or Env]): Specifies the requirements that will be installed, and the environment
            vars that should be attached to the Lambda. Accepts three possible types:\n
            1. A dict which should contain the following keys:\n
               a. reqs: a list of the python libraries, to be installed by the Lambda, or just a
               ``requirements.txt`` string.\n
               b. env_vars: dictionary containing the env_vars that will be a part of the lambda configuration.\n
            2. A list of strings, containing all the required python packages.\n
            3. An instance of Runhouse Env class.\n
            By default, ``runhouse`` package will be installed, and env_vars will include ``{HOME: /tmp/home}``.
        runtime: (Optional[str]): The coding language of the function. Should be one of the following:
            python3.7, python3.8, python3.9, python3.10, python3.11. (Default: ``python3.9``)
        timeout: Optional[int]: The maximum amount of time (in seconds) during which the Lambda will run in AWS
            without timing-out. (Default: ``900``, Min: ``3``, Max: ``900``)
        memory_size: Optional[int], The amount of memory (in MB) to be allocated to the Lambda.
             (Default: ``10240``, Min: ``128``, Max: ``10240``)
        tmp_size: Optional[int], This size of the /tmp folder in the aws lambda file system.
             (Default: ``10240``, Min: ``512``, Max: ``10240``).
        retention_time: Optional[int] The time (in days) the Lambda execution logs will be saved in AWS
            cloudwatch. After that, they will be deleted. (Default: ``30`` days)
        load_from_den (bool): Whether to try loading the Function resource from Den. (Default: ``True``)
        dryrun (bool): Whether to create the Function if it doesn't exist, or load the Function object as a dryrun.
            (Default: ``False``).

    Returns:
        LambdaFunction: The resulting AWS Lambda Function object.

        .. note::
            When creating a Lambda function for the first time (not reloading it), a callable function is a mandatory
            argument.

    Examples:
        >>> import runhouse as rh

        >>> # Pass in a callable function  when creating a Lambda
        >>> def multiply(a, b):
        >>>     return a * b
        >>> multiply_lambda = rh.aws_lambda_fn(fn=multiply, name="lambdas_mult_func")
        >>> mult_res = multiply_lambda(4, 5)  # returns 20.

        >>> # Load function from above
        >>> reloaded_function = rh.aws_lambda_fn(name="lambdas_mult_func")
        >>> reloaded_function_res = reloaded_function(3, 4)  # returns 12.

    """
    # TODO: [SB] in the next phase, maybe add the option to create func from git.
    if name and not any([runtime, fn]):
        # Try reloading existing function
        return LambdaFunction.from_name(
            name=name, load_from_den=load_from_den, dryrun=dryrun
        )

    if not fn or not isinstance(fn, Callable):
        raise RuntimeError(
            "Please provide a callable function OR use from_handler_file method"
            + "in order to create a Lambda function."
        )
    # Env setup.
    env = LambdaFunction._validate_and_create_env(env)

    # extract function pointers, path to code and arg names from callable function.
    handler_function_name = fn.__name__
    fn_pointers = Function._extract_pointers(fn)
    (
        local_path_containing_function,
        should_add,
    ) = Function._get_local_path_containing_module(
        fn_pointers[0], reqs=[] if env is None else env.reqs
    )
    if should_add and env is not None:
        env.reqs = [str(local_path_containing_function)] + env.reqs
    paths_to_code = [extract_module_path(fn)]
    if name is None:
        name = fn.__name__

    # ------- arguments validation -------
    (
        paths_to_code,
        env,
        runtime,
        timeout,
        memory_size,
        tmp_size,
        retention_time,
    ) = LambdaFunction.arguments_validation(
        paths_to_code, env, runtime, timeout, memory_size, tmp_size, retention_time
    )

    new_function = LambdaFunction(
        fn_pointers=fn_pointers,
        paths_to_code=paths_to_code,
        handler_function_name=handler_function_name,
        runtime=runtime,
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
