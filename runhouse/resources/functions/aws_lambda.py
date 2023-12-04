import base64
import contextlib
import copy
import inspect
import json
import logging
import os
import time
import warnings
import zipfile
from pathlib import Path
from typing import Any, Callable, List, Optional

import boto3
import botocore.exceptions

from runhouse.globals import rns_client
from runhouse.resources.envs import _get_env_from, Env

from runhouse.resources.functions.function import Function

logger = logging.getLogger(__name__)

CRED_PATH = f"{Path.home()}/.aws/credentials"
SUPPORTED_RUNTIMES = [
    "python3.7",
    "python3.8",
    "python3.9",
    "python3.10",
    "python 3.11",
]
DEFAULT_PY_VERSION = "python3.9"


class AWSLambdaFunction(Function):
    RESOURCE_TYPE = "lambda_function"
    DEFAULT_ACCESS = "write"
    DEFAULT_ROLE_POLICIES = [
        "cloudwatch:*",
        "lambda:Invoke",
        "lambda:InvokeAsync",
        "lambda:InvokeFunction",
        "lambda:PublishVersion",
        "logs:*",
        "s3:DeleteObject",
        "s3:GetObject",
        "s3:ListBucket",
        "s3:PutObject",
    ]
    MAX_WAIT_TIME = 60  # seconds, max time that can pass before we raise an exception that AWS update takes too long.
    DEFAULT_REGION = "us-east-1"
    DEFAULT_RETENTION = 30  # one month, for lambdas log groups.
    DEFAULT_TIMEOUT = 900  # sec. meaning 15 min.
    DEFAULT_MEMORY_SIZE = 1024  # MB, meaning 1GB.

    # setup needs to be here and not in __init__ for the reload option.
    if Path(CRED_PATH).is_file():
        LAMBDA_CLIENT = boto3.client("lambda")
        LOGS_CLIENT = boto3.client("logs")

    else:
        LAMBDA_CLIENT = boto3.client("lambda", region_name=DEFAULT_REGION)
        LOGS_CLIENT = boto3.client("logs", region_name=DEFAULT_REGION)

    def __init__(
        self,
        paths_to_code: list[str],
        handler_function_name: str,
        runtime: str,
        args_names: list[str],
        fn_pointers: tuple,
        name: str,
        env: Env,
        timeout: int,  # seconds
        memory_size: int,  # MB
        retention_time: int = DEFAULT_RETENTION,  # days
        dryrun: bool = False,
        access: Optional[str] = None,
        **kwargs,  # We have this here to ignore extra arguments when calling from from_config
    ):
        """
        Runhouse AWS Lambda object. It is comprised of the entry point, configuration,
        and dependencies necessary to run the service.

        .. note::
                To create an AWS lambda resource, please use the factory method :func:`aws_lambda_function`.
        """

        super().__init__(
            name=name,
            dryrun=dryrun,
            system=self.RESOURCE_TYPE,
            env=env,
            fn_pointers=fn_pointers,
            access=access or self.DEFAULT_ACCESS,
            **kwargs,
        )

        self.local_path_to_code = paths_to_code
        self.handler_function_name = handler_function_name
        self.args_names = args_names
        self.runtime = runtime
        self.timeout = timeout
        self.memory_size = memory_size
        self.aws_lambda_config = (
            None  # Lambda config and role arn from aws will be saved here
        )
        self.retention_time = retention_time

    # ----------------- Constructor helper methods -----------------

    @classmethod
    def from_config(cls, config: dict, dryrun: bool = False):
        """Create an AWS Lambda object from a config dictionary."""

        if "resource_subtype" in config.keys():
            config.pop("resource_subtype", None)
        if "system" in config.keys():
            config.pop("system", None)
        keys = config.keys()
        if "fn_pointers" not in keys:
            config["fn_pointers"] = None
        if "timeout" not in keys:
            config["timeout"] = AWSLambdaFunction.DEFAULT_TIMEOUT
        if "memory_size" not in keys:
            config["memory_size"] = AWSLambdaFunction.DEFAULT_MEMORY_SIZE
        if "env" not in keys:
            config["env"] = Env(reqs=[], name=Env.DEFAULT_NAME)
        else:
            config["env"] = Env.from_config(config["env"])

        return AWSLambdaFunction(**config, dryrun=dryrun).to()

    @classmethod
    def from_name(cls, name, dryrun=False, alt_options=None):
        funcs = cls.LAMBDA_CLIENT.list_functions()
        funcs = funcs["Functions"]
        func_names = [func["FunctionName"] for func in funcs]
        if name in func_names:
            func = super().from_name(name=name)
            return func.to()
        else:
            raise RuntimeError(
                f"Could not find a Lambda called {name}. Please provide a name of an existing Lambda, "
                + "or paths_to_code, handler_function_name, runtime and args_names (and a name if you"
                + " wish), in order to create a new lambda."
            )

    # ----------------- Private helping methods -----------------

    @classmethod
    def _check_for_child_configs(cls, config):
        """Overload this method of the function class."""
        return config

    @classmethod
    def _reqs_to_list(cls, env):
        """Converting requirements from requirements.txt to a list"""
        if env == "requirements.txt":
            env = Path(env).absolute()

        def _get_lib_name(req: str):
            index = len(req)
            if "<" in req:
                index = req.index("<")
            elif ">" in req:
                index = req.index(">")
            elif "=" in req:
                index = req.index("=")
            return req[:index]

        with open(env, "r") as f:
            reqs = f.read().split("\n")
            reqs = [_get_lib_name(req) for req in reqs]
        return reqs

    def _lambda_exist(self, name):
        """Checks if a Lambda with the name given during init is already exists in AWS"""
        func_names = [
            func["FunctionName"]
            for func in self.LAMBDA_CLIENT.list_functions()["Functions"]
        ]
        return name in func_names

    @contextlib.contextmanager
    def _wait_until_lambda_update_is_finished(self, name):
        """Verifies that a running update of the function (in AWS) is finished (so the next one could be executed)"""

        time_passed = 0  # seconds
        response = self.LAMBDA_CLIENT.get_function(FunctionName=name)
        state = response["Configuration"]["State"] == "Active"

        try:
            last_update_status = (
                response["Configuration"]["LastUpdateStatus"] != "InProgress"
            )
        except KeyError:
            last_update_status = False

        while True:
            if state and last_update_status:
                break
            else:
                if time_passed > self.MAX_WAIT_TIME:
                    raise TimeoutError(
                        f"Lambda function called {name} is being deployed in AWS for too long. "
                        + " Please check the resource in AWS console, delete relevant resource(s) "
                        + "if necessary, and re-run your Runhouse code."
                    )
                time.sleep(1)
                time_passed += 1
                response = self.LAMBDA_CLIENT.get_function(FunctionName=name)
                state = response["Configuration"]["State"] == "Active"
                try:
                    last_update_status = (
                        response["Configuration"]["LastUpdateStatus"] != "InProgress"
                    )
                except KeyError:
                    last_update_status = False

        yield True

    def _rh_wrapper(self):
        """Creates a runhouse wrapper to the handler function"""
        handler_path = self.local_path_to_code[0]
        wrapper_path = str(Path(handler_path).parent / f"rh_handler_{self.name}.py")
        handler_name = Path(handler_path).stem
        Path(handler_path).touch()

        f = open(wrapper_path, "w")
        f.write("import subprocess\n" "import sys\n" "import os\n\n")
        f.write("def lambda_handler(event, context):\n")

        # adding code for installing python libraries
        reqs = self.env.reqs
        # TODO: checking the posability to change the home dir
        # if "runhouse" not in reqs:
        #     reqs.append("runhouse")
        if "./" in reqs:
            reqs.remove("./")
        for req in reqs:
            f.write(
                f"\tif not os.path.isdir('/tmp/{req}'):\n"
                f"\t\tsubprocess.call(['pip', 'install', '{req}', '-t', '/tmp/'])\n"
            )
        if reqs is not None and len(reqs) > 0:
            f.write("\tsys.path.insert(1, '/tmp/')\n\n")

        f.write(
            f"\tfrom {handler_name} import {self.handler_function_name}\n"
            f"\treturn {self.handler_function_name}(**event)"
        )
        f.close()
        return wrapper_path

    @classmethod
    def _supported_python_libs(cls):
        """Returns a list of the supported python libs by the AWS Lambda resource"""
        # TODO [SB]: think what is the better implementation: via website, or AWS lambda. For now it is hard-coded.
        supported_libs = [
            "urllib3",
            "six",
            "simplejson",
            "s3transfer",
            "python-dateutil",
            "jmespath",
            "botocore",
            "boto3",
            "awslambdaric",
            "setuptools",
            "pip",
        ]

        return supported_libs

    def _update_lambda_config(self, env_vars):
        """Updates existing Lambda in AWS (config) that was provided in the init."""
        logger.info(f"Updating a Lambda called {self.name}")
        lambda_config = self.LAMBDA_CLIENT.update_function_configuration(
            FunctionName=self.name,
            Runtime=self.runtime,
            Timeout=self.timeout,
            MemorySize=self.memory_size,
            Environment={"Variables": env_vars},
        )

        # TODO [SB]: in the next phase, enable for other to update the Lambda code.
        # wait for the config update process to finish, and then update the code (Lambda logic).
        with self._wait_until_lambda_update_is_finished(self.name):
            path = str(Path(self.local_path_to_code[0]).absolute()).split("/")
            path = ["/" + path_e for path_e in path]
            path[0] = ""
            zip_file_name = f"{''.join(path[:-1])}/{self.name}_code_files.zip"
            zf = zipfile.ZipFile(zip_file_name, mode="w")
            try:
                for file_name in self.local_path_to_code:
                    zf.write(file_name, str(Path(file_name).name))

            except FileNotFoundError:
                logger.error(f"Could not find {FileNotFoundError.filename}")
            finally:
                zf.close()
            with open(zip_file_name, "rb") as f:
                zipped_code = f.read()

            lambda_config = self.LAMBDA_CLIENT.update_function_code(
                FunctionName=self.name, ZipFile=zipped_code
            )

        logger.info(f'{lambda_config["FunctionName"]} was updated successfully.')

        return lambda_config

    def _deploy_lambda_to_aws_helper(self, role_res, zipped_code, env_vars):
        """Deploying new lambda to AWS helping function"""
        try:
            lambda_config = self.LAMBDA_CLIENT.create_function(
                FunctionName=self.name,
                Runtime=self.runtime,
                Role=role_res["Role"]["Arn"],
                Handler=f"rh_handler_{self.name}.lambda_handler",
                Code={"ZipFile": zipped_code},
                Timeout=self.timeout,
                MemorySize=self.memory_size,
                Environment={"Variables": env_vars},
                EphemeralStorage={
                    "Size": self.DEFAULT_MEMORY_SIZE
                },  # size of /tmp folder.
            )
            func_name = lambda_config["FunctionName"]
            with self._wait_until_lambda_update_is_finished(func_name):
                lambda_config = self.LAMBDA_CLIENT.get_function(FunctionName=func_name)
                return lambda_config
        except self.LAMBDA_CLIENT.exceptions.InvalidParameterValueException or botocore.exceptions.ClientError:
            return False

    @contextlib.contextmanager
    def _deploy_lambda_to_aws(self, role_res, zipped_code, env_vars):
        """Deploying new lambda to AWS"""
        config = self._deploy_lambda_to_aws_helper(role_res, zipped_code, env_vars)
        time_passed = 0
        while config is False:
            if time_passed > self.MAX_WAIT_TIME:
                raise TimeoutError(
                    f"Role called {role_res['RoleName']} is being created AWS for too long."
                    + " Please check the resource in AWS console, delete relevant resource(s) "
                    + "if necessary, and re-run your Runhouse code."
                )
            time_passed += 1
            time.sleep(1)
            config = self._deploy_lambda_to_aws_helper(role_res, zipped_code, env_vars)
        yield config

    def _create_role_in_aws(self):
        """create a new role for the lambda function. If already exists - returns it."""

        iam_client = boto3.client("iam")

        role_policy = {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Action": self.DEFAULT_ROLE_POLICIES,
                    "Resource": "*",
                    "Effect": "Allow",
                }
            ],
        }

        assume_role_policy_document = {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Principal": {"Service": "lambda.amazonaws.com"},
                    "Action": "sts:AssumeRole",
                }
            ],
        }

        try:
            role_res = iam_client.create_role(
                RoleName=f"{self.name}_Role",
                AssumeRolePolicyDocument=json.dumps(assume_role_policy_document),
            )
            logger.info(f'{role_res["Role"]["RoleName"]} was created successfully.')

        except iam_client.exceptions.EntityAlreadyExistsException:
            role_res = iam_client.get_role(RoleName=f"{self.name}_Role")

        iam_client.put_role_policy(
            RoleName=role_res["Role"]["RoleName"],
            PolicyName=f"{self.name}_Policy",
            PolicyDocument=json.dumps(role_policy),
        )

        return role_res

    def _create_new_lambda(self, env_vars):
        """Creates new AWS Lambda."""
        logger.info(f"Creating a new Lambda called {self.name}")
        path = str(Path(self.local_path_to_code[0]).absolute()).split("/")
        path = ["/" + path_e for path_e in path]
        path[0] = ""
        zip_file_name = f"{''.join(path[:-1])}/{self.name}_code_files.zip"
        zf = zipfile.ZipFile(zip_file_name, mode="w")
        try:
            for file_name in self.local_path_to_code:
                zf.write(file_name, str(Path(file_name).name))
        except FileNotFoundError:
            logger.error(f"Could not find {FileNotFoundError.filename}")
        finally:
            zf.close()
        with open(zip_file_name, "rb") as f:
            zipped_code = f.read()

        # creating a role for the Lambda, using default policy.
        # TODO [SB]: in the next phase, enable the user to update the default policy
        role_res = self._create_role_in_aws()

        with self._deploy_lambda_to_aws(
            role_res, zipped_code, env_vars
        ) as lambda_config:
            logger.info(
                f'{lambda_config["Configuration"]["FunctionName"]} was created successfully.'
            )
            Path(zip_file_name).absolute().unlink()
            return lambda_config

    def to(
        self,
        # Variables below are deprecated
        env: Optional[List[str]] = [],
        reqs: Optional[List[str]] = None,
        setup_cmds: Optional[List[str]] = [],
        force_install: Optional[bool] = False,
        cloud: str = "aws_lambda",
    ):
        """
        Set up a function on AWS as a Lambda function.

        See the args of the factory method :func:`aws_lambda` for more information.

        Example:
            >>> my_aws_lambda = rh.aws_lambda_function(path_to_codes=["full/path/to/model_a_handler.py"],
            >>> handler_function_name='main_func',
            >>> runtime='python_3_9',
            >>> name="my_lambda_func").to()
        """

        # Checking if the user have a credentials file
        if not Path(CRED_PATH).is_file():
            logger.error("No credentials found")
            raise FileNotFoundError("No credentials found")

        rh_handler_wrapper = self._rh_wrapper()
        self.local_path_to_code.append(rh_handler_wrapper)

        env_vars = self.env.env_vars if self.env else {}

        # if function exist - will update it. Else, a new one will be created.
        if self._lambda_exist(self.name):
            # updating the configuration with the initial configuration.
            # TODO [SB]: in the next phase, enable the user to change the config of the Lambda.
            # lambda_config = self._update_lambda_config(env_vars)
            lambda_config = self.LAMBDA_CLIENT.get_function(FunctionName=self.name)

        else:
            # creating a new Lambda function, since it's not existing in the AWS account which is configured locally.
            lambda_config = self._create_new_lambda(env_vars)

        self.aws_lambda_config = lambda_config
        Path(rh_handler_wrapper).absolute().unlink()
        return self

    #
    #     # ----------------- Function call methods -----------------
    #

    def __call__(self, *args, **kwargs) -> Any:
        """Call (invoke) the Lambdas function

        Args:
             *args: Optional args for the Function
             **kwargs: Optional kwargs for the Function

        Returns:
            The Function's return value
        """
        return self._invoke(*args, **kwargs)

    def _get_log_group_names(self, log_group_prefix):
        lambdas_log_groups = self.LOGS_CLIENT.describe_log_groups(
            logGroupNamePrefix=log_group_prefix
        )["logGroups"]
        lambdas_log_groups = [group["logGroupName"] for group in lambdas_log_groups]
        return lambdas_log_groups

    @contextlib.contextmanager
    def _wait_till_log_group_is_created(
        self, curr_lambda_log_group, lambdas_log_groups, log_group_prefix
    ):
        time_passed = 0
        while curr_lambda_log_group not in lambdas_log_groups:
            if time_passed > self.MAX_WAIT_TIME:
                raise TimeoutError(
                    f"The log group called {log_group_prefix} is being deployed in AWS cloudwatch for too long. "
                    + " Please check the resource in AWS console, delete relevant resource(s) "
                    + "if necessary, and re-run your Runhouse code."
                )
            time.sleep(1)
            lambdas_log_groups = self._get_log_group_names(log_group_prefix)
            time_passed += 1
        yield True

    def _invoke(self, *args, **kwargs) -> Any:
        log_group_prefix = "/aws/lambda/"
        lambdas_log_groups = self._get_log_group_names(log_group_prefix)
        curr_lambda_log_group = f"{log_group_prefix}{self.name}"
        invoke_for_the_first_time = curr_lambda_log_group not in lambdas_log_groups

        payload_invoke = {}

        if len(args) > 0 and self.args_names is not None:
            payload_invoke = {self.args_names[i]: args[i] for i in range(len(args))}
        invoke_res = self.LAMBDA_CLIENT.invoke(
            FunctionName=self.name,
            Payload=json.dumps({**payload_invoke, **kwargs}),
            LogType="Tail",
        )
        return_value = invoke_res["Payload"].read().decode("utf-8")

        try:
            logger.error(invoke_res["FunctionError"])
            return_value = json.loads(return_value)
            raise RuntimeError(
                f"Failed to run {self.name}. Error: {return_value['errorType']}, {return_value['errorMessage']}"
            )

        except KeyError:
            if invoke_for_the_first_time:
                lambdas_log_groups = self._get_log_group_names(log_group_prefix)
                with self._wait_till_log_group_is_created(
                    curr_lambda_log_group, lambdas_log_groups, log_group_prefix
                ):
                    self.LOGS_CLIENT.put_retention_policy(
                        logGroupName=curr_lambda_log_group,
                        retentionInDays=self.retention_time,
                    )

            log_lines = "Function Logs are:\n" + base64.b64decode(
                invoke_res["LogResult"]
            ).decode("utf-8")

            print(log_lines)
            return return_value

    def map(self, *args, **kwargs):
        """Map a function over a list of arguments.

        Example:
            >>> # The my_lambda_handler.py file
            >>> def my_summer(arg1, arg2, arg3):
            >>>     return arg1 + arg2 + arg3
            >>>
            >>> # your 'main' python file, where you are using runhouse
            >>> aws_lambda = rh.aws_lambda_function(path_to_code=["full/path/to/my_lambda_handler.py"],
            >>> handler_function_name='my_summer',
            >>> runtime='python_3_9',
            >>> name="my_summer").to()
            >>> aws_lambda.map([1, 2], [1, 4], [2, 3])
            >>> # output: ["4", "9"] (It returns str type because of AWS API)

        """

        return [self._invoke(*args, **kwargs) for args in zip(*args)]

    #
    def starmap(self, args_lists, **kwargs):
        """Like :func:`map` except that the elements of the iterable are expected to be iterables
        that are unpacked as arguments. An iterable of [(1,2), (3, 4)] results in [func(1,2), func(3,4)].

        Example:
            >>> arg_list = [(1,2), (3, 4)]
            >>> # runs the function twice, once with args (1, 2) and once with args (3, 4)
            >>> my_aws_lambda.starmap(arg_list)
        """

        return [self._invoke(*args, **kwargs) for args in args_lists]

    # ----------------- Properties setup -----------------
    @property
    def config_for_rns(self):
        config = super().config_for_rns
        config.update(
            {
                "paths_to_code": self.local_path_to_code,
                "handler_function_name": self.handler_function_name,
                "runtime": self.runtime,
                "timeout": self.timeout,
                "memory_size": self.memory_size,
                "args_names": self.args_names,
            }
        )
        return config

    @property
    def handler_path(self):
        return self.local_path_to_code[0]


def _paths_to_code_from_fn_pointers(fn_pointers):
    """creates path to code from fn_pinters"""
    root_dir = rns_client.locate_working_dir()
    file_path = fn_pointers[1].replace(".", "/") + ".py"
    paths_to_code = [os.path.join(root_dir, file_path)]
    return paths_to_code


def aws_lambda_function(
    fn: Callable = None,
    paths_to_code: list[str] = None,
    handler_function_name: str = None,
    runtime: str = None,
    args_names: list[str] = None,
    name: Optional[str] = None,
    env: Optional[dict or list[str] or Env] = None,
    timeout: Optional[int] = None,
    memory_size: Optional[int] = None,
    region: Optional[str] = None,
    retention_time: Optional[int] = None,
    dryrun: bool = False,
):
    """Builds an instance of :class:`AWSLambdaFunction`.

    Args:
        fn (Optional[Callable]): The Lambda function to be executed.
        paths_to_code: (Optional[list[str]]): List of the FULL paths to the python code file(s) that should be sent to
            AWS Lambda. First path in the list should be the path to the handler file which contaitns the main
            (handler) function. If ``fn`` is provided, this argument is ignored.
        handler_function_name: str: The name of the function in the handler file that will be executed by the Lambda.
        runtime: str: The coding language of the fuction. Should be one of the following:
            python3.7, python3.8, python3.9, python3.10, python 3.11. (Default: ``python3.9``)
        args_names: (Optional[list[str]]): List of the function's accepted parameters, which will be passed to the
            Lambda Function. If ``fn`` is provided, this argument is ignored.
            If your function doesn't accept arguments, please provide an empty list.
        name (Optional[str]): Name of the Lambda Function to create or retrieve.
            This can be either from a local config or from the RNS.
        env (Optional[Dict or List[str] or Env]): Specifies the requirments that will be installed, and the enviourment
            vars that should be attached to the Lambda. Excepts three possible types:
            1. A dict which should contain the following keys:
            reqs - a lisr og the python libraries, which will be used by the Lambda or just a 'requiremnts.txt' string.
            env_vars: dictionary containing the env_vars that will be a part of the lambda configuration.
            2. A list of strings, containing all the required python packeages.
            3. An insrantce of Runhouse Env class.
            By default, ``runhouse`` package will be installed.
        timeout: Optional[int]: The maximum amount of time (in secods) during which the Lambda will run in AWS
            without timing-out. (Default: ``900``, Min: ``3``, Max: ``900``)
        memory_size: Optional[int], The amount of memeory, im MB, that will be aloocatied to the lambda.
             (Default: ``128``, Min: ``128``, Max: ``10240``)
        region: The AWS region where the lambda will be created. In order to make sure that any packages will be
            installed proparly, please provide the closest region to your workplace. (Default: ``us-east-1``)
        retention_time: Optional[int] The time (in days) that sepcific lambda exection logs will be saved in AWS
            cloudwatch. After that, they will be deleted. (Default: ``30`` days)
        dryrun (bool): Whether to create the Function if it doesn't exist, or load the Function object as a dryrun.
            (Default: ``False``). Is not used by Lambda, but is a port of Function constructor signatuere.

    Returns:
        AWSLambdaFunction: The resulting AWS Lambda Function object.

        .. note::
            When creating the function for the first time (and not reloading it), the following arguments are
            mandatory: paths_to_code, handler_function_name, runtime, args_names.

    Example:
        >>> import runhouse as rh

        >>> # handler_file.py
        >>> def summer(a, b):
        >>>    return a + b

        >>> # your 'main' python file, where you are using runhouse
        >>> lambdas_func = rh.aws_lambda_function(
        >>>                     paths_to_code=['/full/path/to/handler_file.py'],
        >>>                     handler_function_name = 'summer',
        >>>                     runtime = 'python3.9',
        >>>                     name="my_func").to().save()

        >>> # using the function
        >>> res = summer(5, 8)  # returns "13". (It returns str type because of AWS API)

        >>> # Load function from above
        >>> reloaded_function = rh.aws_lambda_function(name="my_func")

        >>> # Pass in the function itself when creating the Lambda
        >>> lambdas_func = rh.aws_lambda_function(fn=summer, name="lambdas_func")

    """
    if name and not any(
        [paths_to_code, handler_function_name, runtime, fn, args_names]
    ):
        # Try reloading existing function
        return AWSLambdaFunction.from_name(name=name)

    # TODO: [SB] in the next phase, maybe add the option to create func from git.

    if env is not None and not isinstance(env, Env):
        original_env = copy.deepcopy(env)
        if isinstance(original_env, dict) and "env_vars" in original_env.keys():
            env = _get_env_from(env["reqs"]) or Env(
                working_dir="../serverless/", name=Env.DEFAULT_NAME
            )
            env.env_vars = (
                original_env["env_vars"] if original_env["env_vars"] is not None else {}
            )
        elif isinstance(original_env, str):
            env = _get_env_from(AWSLambdaFunction._reqs_to_list(env))
        else:
            env = _get_env_from(env) or Env(
                working_dir="../serverless/", name=Env.DEFAULT_NAME
            )

    elif env is None:
        env = Env(reqs=[], name=Env.DEFAULT_NAME)

    if isinstance(fn, Callable):
        handler_function_name = fn.__name__
        fn_pointers = Function._extract_pointers(
            fn, reqs=[] if env is None else env.reqs
        )
        paths_to_code = _paths_to_code_from_fn_pointers(fn_pointers)
        args_names = [param.name for param in inspect.signature(fn).parameters.values()]
        if name is None:
            name = fn.__name__
    else:

        # ------- arguments validation -------
        if paths_to_code is None or len(paths_to_code) == 0:
            raise RuntimeError("Please provide a path to the lambda handler file.")
        if handler_function_name is None or len(handler_function_name) == 0:
            raise RuntimeError(
                "Please provide the name of the function that should be executed by the lambda."
            )
        if args_names is None:
            # Parsing the handler file and extracting the arguments names of the handler function.
            file_path = paths_to_code[0]
            func_name = handler_function_name[0]
            with (open(file_path) as f):
                a = f.read()
                index = a.index(func_name)
                open_par = index + len(func_name)
                close_par = open_par + 1
                while a[close_par] != ")":
                    close_par += 1
                args_names = a[open_par + 1 : close_par].split(",")
                args_names = [
                    arg.strip().split(":")[0] for arg in args_names if len(arg) > 0
                ]
            warnings.warn(
                f"Arguments names were not provided. Extracted the following args names: {args_names}."
            )
        if name is None:
            name = handler_function_name.replace(".", "_")
        fn_pointers = None

    # ------- More arguments validation -------
    if runtime is None or runtime not in SUPPORTED_RUNTIMES:
        warnings.warn(
            f"{runtime} is not a supported by AWS Lambda. Setting runtime to python3.9."
        )
        runtime = DEFAULT_PY_VERSION

    if timeout is None:
        warnings.warn("Timeout set to 15 min.")
        timeout = AWSLambdaFunction.DEFAULT_TIMEOUT
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
        memory_size = AWSLambdaFunction.DEFAULT_MEMORY_SIZE
    else:
        if memory_size < 128:
            memory_size = 128
            warnings.warn("Memory size can not be less then 128 MB, setting to 128 MB.")
        if memory_size > 10240:
            memory_size = 10240
            warnings.warn(
                "Memory size can not be more then 10240 MB, setting to 10240 MB."
            )
    if region is None:
        region = AWSLambdaFunction.DEFAULT_REGION
    if retention_time is None:
        retention_time = AWSLambdaFunction.DEFAULT_RETENTION

    new_function = AWSLambdaFunction(
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
        region=region,
        retention_time=retention_time,
    ).to()

    return new_function
