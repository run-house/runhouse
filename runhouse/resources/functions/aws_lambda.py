import base64
import contextlib
import copy
import importlib
import inspect
import json
import time
import warnings
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

try:
    import boto3
    import botocore.exceptions
except ImportError:
    pass

from runhouse.globals import rns_client
from runhouse.logger import get_logger
from runhouse.resources.envs import _get_env_from, Env
from runhouse.resources.functions.function import Function


CRED_PATH = f"{Path.home()}/.aws/credentials"
LOG_GROUP_PREFIX = "/aws/lambda/"

logger = get_logger(__name__)


class LambdaFunction(Function):
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
        "kms:Decrypt",
    ]
    MAX_WAIT_TIME = 60  # seconds, max time that can pass before we raise an exception that AWS update takes too long.
    DEFAULT_REGION = "us-east-1"
    DEFAULT_RETENTION = 30  # one month, for lambdas log groups.
    DEFAULT_TIMEOUT = 900  # sec. meaning 15 min.
    DEFAULT_MEMORY_SIZE = 10240  # MB
    DEFAULT_TMP_SIZE = 10240  # MB, meaning 10G
    HOME_DIR = "/tmp/home"
    SUPPORTED_RUNTIMES = [
        "python3.7",
        "python3.8",
        "python3.9",
        "python3.10",
        "python3.11",
    ]
    DEFAULT_PY_VERSION = "python3.9"

    def __init__(
        self,
        paths_to_code: List[str],
        handler_function_name: str,
        runtime: str,
        fn_pointers: tuple,
        name: str,
        env: Env,
        timeout: int,  # seconds
        memory_size: int,  # MB
        tmp_size: int,  # MB
        retention_time: int = DEFAULT_RETENTION,  # days
        dryrun: bool = False,
        access: Optional[str] = None,
        **kwargs,  # We have this here to ignore extra arguments when calling from from_config
    ):
        """
        Runhouse AWS Lambda object. It is comprised of the entry point, configuration,
        and dependencies necessary to run the service on AWS infra.

        .. note::
                To create an AWS lambda resource, please use the factory method :func:`aws_lambda_fn`.
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
        self.runtime = runtime
        self.timeout = timeout
        self.memory_size = memory_size
        self.tmp_size = tmp_size
        self.aws_lambda_config = (
            None  # Lambda config and role arn from aws will be saved here
        )
        self.retention_time = retention_time
        self.args_names = LambdaFunction.extract_args_from_file(
            paths_to_code, handler_function_name
        )

        # set-up in order to prevent read timeout during lambda invocations
        config = botocore.config.Config(read_timeout=900, connect_timeout=900)

        # will be used for reloading shared functions from other regions.
        function_arn = (
            kwargs.get("function_arn") if "function_arn" in kwargs.keys() else None
        )
        if function_arn:
            region = function_arn.split(":")[3]
            self.lambda_client = boto3.client(
                "lambda", region_name=region, config=config
            )
            self.logs_client = boto3.client("logs", region_name=region)
        else:

            if Path(CRED_PATH).is_file():
                self.lambda_client = boto3.client("lambda", config=config)
                self.logs_client = boto3.client("logs")

            else:
                self.lambda_client = boto3.client(
                    "lambda", region_name=self.DEFAULT_REGION
                )
                self.logs_client = boto3.client("logs", region_name=self.DEFAULT_REGION)
        self.iam_client = boto3.client("iam")

    # --------------------------------------
    # Constructor helper methods
    # --------------------------------------

    @classmethod
    def from_config(
        cls, config: dict, dryrun: bool = False, _resolve_children: bool = True
    ):
        if "resource_subtype" in config.keys():
            config.pop("resource_subtype", None)
        if "system" in config.keys():
            config.pop("system", None)
        keys = config.keys()
        if "fn_pointers" not in keys:
            config["fn_pointers"] = None
        if "timeout" not in keys:
            config["timeout"] = LambdaFunction.DEFAULT_TIMEOUT
        if "memory_size" not in keys:
            config["memory_size"] = LambdaFunction.DEFAULT_MEMORY_SIZE
        if "env" not in keys:
            config["env"] = Env(
                reqs=[],
                env_vars={"HOME": LambdaFunction.HOME_DIR},
            )
        else:
            config["env"] = Env.from_config(
                config["env"], _resolve_children=_resolve_children
            )

        return LambdaFunction(**config, dryrun=dryrun).deploy()

    @classmethod
    def from_name(
        cls,
        name,
        load_from_den: bool = True,
        dryrun: bool = False,
        _alt_options: Dict = None,
        _resolve_children: bool = True,
    ):
        config = rns_client.load_config(name=name, load_from_den=load_from_den)
        if not config:
            raise ValueError(f"Could not find a Lambda called {name}.")
        return cls.from_config(config, _resolve_children=_resolve_children)

    @classmethod
    def from_handler_file(
        cls,
        paths_to_code: List[str],
        handler_function_name: str,
        name: Optional[str] = None,
        env: Optional[Union[Dict, List[str], Env]] = None,
        runtime: Optional[str] = None,
        timeout: Optional[int] = None,
        memory_size: Optional[int] = None,
        tmp_size: Optional[int] = None,
        retention_time: Optional[int] = None,
        dryrun: bool = False,
    ):
        """
        Creates an AWS Lambda function from a Python file.

        Args:
            paths_to_code: (Optional[List[str]]): List of the FULL paths to the Python code file(s) that should be
                sent to AWS Lambda. First path in the list should be the path to the handler file which contains
                the main (handler) function. If ``fn`` is provided, this argument is ignored.
            handler_function_name: (Optional[str]): The name of the function in the handler file that will be executed
                by the Lambda. If ``fn`` is provided, this argument is ignored.
            runtime: (Optional[str]): The coding language of the function. Should be one of the following:
                python3.7, python3.8, python3.9, python3.10, python3.11. (Default: ``python3.9``)
            name (Optional[str]): Name of the Lambda Function to create or retrieve.
                This can be either from a local config or from the RNS.
            env (Optional[Dict or List[str] or Env]): Specifies the requirements that will be installed, and the
                environment vars that should be attached to the Lambda. Accepts three possible types:

                1. A dict which should contain the following keys:

                    reqs: a list of the python libraries, to be installed by the Lambda, or just a
                    ``requirements.txt``string.

                    env_vars: dictionary containing the env_vars that will be a part of the lambda configuration.

                2. A list of strings, containing all the required python packages.

                3. An instance of Runhouse Env class. By default, ``runhouse`` package will be installed, and env_vars
                   will include ``{HOME: /tmp/home}``.

            timeout (Optional[int]) The maximum amount of time (in seconds) during which the Lambda will run in AWS
                without timing-out. (Default: ``900``, Min: ``3``, Max: ``900``)
            memory_size (Optional[int]), The amount of memory (in MB) to be allocated to the Lambda.
                (Default: ``10240``, Min: ``128``, Max: ``10240``)
            tmp_size (Optional[int]), This size of the /tmp folder in the aws lambda file system.
                (Default: ``10240``, Min: ``512``, Max: ``10240``).
            retention_time (Optional[int]) The time (in days) the Lambda execution logs will be saved in AWS
                cloudwatch. After that, they will be deleted. (Default: ``30`` days)
            dryrun (bool): Whether to create the Function if it doesn't exist, or load the Function object as a dryrun.
                (Default: ``False``).

        Returns:
            LambdaFunction: The resulting AWS Lambda Function object.

        Example:
            >>> # handler_file.py
            >>> def summer(a, b):
            >>>    return a + b

            >>> # your 'main' python file, where you are using runhouse
            >>> summer_lambda = rh.LambdaFunction.from_handler_file(
            >>>                     paths_to_code=['/full/path/to/handler_file.py'],
            >>>                     handler_function_name = 'summer',
            >>>                     runtime = 'python3.9',
            >>>                     name="my_func").save()

            >>> # invoking the function
            >>> summer_res = summer_lambda(5, 8)  # returns 13.
        """

        if name is None:
            name = handler_function_name.replace(".", "_")

        env = LambdaFunction._validate_and_create_env(env)
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
            fn_pointers=None,
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

    # --------------------------------------
    # Private helping methods
    # --------------------------------------

    # Arguments validation and arguments creation methods
    @classmethod
    def _validate_and_create_env(cls, env: Optional[Env] = None):
        """
        Validates the passed env argument, and creates a Runhouse env instance if needed.

        Args:
            env (Optional[Env]): The environment to validate and create.

        Returns:
            Env: a Runhouse env object
        """
        if env is not None and not isinstance(env, Env):
            original_env = copy.deepcopy(env)
            if isinstance(original_env, dict) and "env_vars" in original_env.keys():
                env = _get_env_from(env["reqs"]) or Env(working_dir="./")
            elif isinstance(original_env, str):
                env = _get_env_from(env)
            else:
                env = _get_env_from(env) or Env(working_dir="./")

        if env is None:
            env = Env(
                reqs=[],
                env_vars={"HOME": cls.HOME_DIR},
            )

        if isinstance(env, Env) and "HOME" not in env.env_vars.keys():
            env.env_vars["HOME"] = cls.HOME_DIR

        return env

    @classmethod
    def extract_args_from_file(cls, paths_to_code, handler_function_name):
        file_path = paths_to_code[0]
        name = file_path.split("/")[-1].split(".")[0]

        spec = importlib.util.spec_from_file_location(name, file_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # Extract the function and its arguments
        func = getattr(module, handler_function_name, None)
        args_names = inspect.getfullargspec(func).args
        warnings.warn(
            f"Arguments names were not provided. Extracted the following args names: {args_names}."
        )
        return args_names

    @classmethod
    def arguments_validation(
        cls, paths_to_code, env, runtime, timeout, memory_size, tmp_size, retention_time
    ):

        if isinstance(env, str) and "requirements.txt" in env:
            paths_to_code.append(Path(env).absolute())

        if runtime is None or runtime not in LambdaFunction.SUPPORTED_RUNTIMES:
            warnings.warn(
                f"{runtime} is not a supported by AWS Lambda. Setting runtime to python3.9."
            )
            runtime = LambdaFunction.DEFAULT_PY_VERSION

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
                warnings.warn(
                    "Timeout can not be more then 900 sec, setting to 900 sec."
                )
            if timeout < 3:
                timeout = 3
                warnings.warn("Timeout can not be less then 3 sec, setting to 3 sec.")
        if memory_size is None:
            warnings.warn("Memory size set to 10 GB.")
            memory_size = LambdaFunction.DEFAULT_MEMORY_SIZE
        else:
            if (
                env.reqs is not None or len(env.reqs) > 0
            ) and memory_size < LambdaFunction.DEFAULT_MEMORY_SIZE:
                warnings.warn(
                    "Increasing the memory size to 10240 MB, in order to enable the packages setup."
                )
                memory_size = LambdaFunction.DEFAULT_MEMORY_SIZE
            if memory_size < 128:
                memory_size = 128
                warnings.warn(
                    "Memory size can not be less then 128 MB, setting to 128 MB."
                )
            if memory_size > 10240:
                memory_size = 10240
                warnings.warn(
                    "Memory size can not be more then 10240 MB, setting to 10240 MB."
                )
        if tmp_size is None or tmp_size < LambdaFunction.DEFAULT_TMP_SIZE:
            warnings.warn(
                "Setting /tmp size to 10GB, in order to enable the packages setup."
            )
            tmp_size = LambdaFunction.DEFAULT_TMP_SIZE
        elif tmp_size > 10240:
            tmp_size = 10240
            warnings.warn(
                "/tmp size can not be more then 10240 MB, setting to 10240 MB."
            )
        if retention_time is None:
            retention_time = LambdaFunction.DEFAULT_RETENTION

        return (
            paths_to_code,
            env,
            runtime,
            timeout,
            memory_size,
            tmp_size,
            retention_time,
        )

    def _lambda_exist(self, name):
        """Checks if a Lambda with the name given during init is already exists in AWS"""
        try:
            aws_lambda = self.lambda_client.get_function(FunctionName=name)
            return aws_lambda
        except self.lambda_client.exceptions.ResourceNotFoundException:
            return False

    @contextlib.contextmanager
    def _wait_until_lambda_update_is_finished(self, name):
        """Verifies that a running update of the function (in AWS) is finished (so the next one could be executed)"""

        time_passed = 0  # seconds
        response = self.lambda_client.get_function(FunctionName=name)
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
                response = self.lambda_client.get_function(FunctionName=name)
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
        f.write(f"\tif not os.path.isdir('{self.HOME_DIR}'):\n")
        f.write(f"\t\tos.mkdir('{self.HOME_DIR}')\n")

        # adding code for installing python libraries
        if isinstance(self.env, str):
            f.write(
                "\tsubprocess.call(['pip', 'install', '-r', 'requirements.txt',"
                + " '--ignore-installed', '-t', '{self.HOME_DIR}/'])\n"
                f"\tif not os.path.isdir('{self.HOME_DIR}/runhouse'):\n"
                f"\t\tsubprocess.call(['pip', 'install', 'runhouse', '-t', '{self.HOME_DIR}/'])\n"
                f"\tsys.path.insert(1, '{self.HOME_DIR}/')\n\n"
            )
        else:
            reqs = self.env.reqs
            if "runhouse" not in reqs:
                reqs.append("runhouse")
            if "./" in reqs:
                reqs.remove("./")
            for req in reqs:
                req = req.split()
                dir_name = req[0]
                if len(req) > 1:
                    req = "".join("'%s', " % str(s) for s in req)[:-1]
                else:
                    req = f"'{req[0]}',"
                f.write(
                    f"\tif not os.path.isdir('{self.HOME_DIR}/{dir_name}'):\n"
                    f"\t\tsubprocess.call(['pip', 'install', {req} '-t', '{self.HOME_DIR}/'])\n"
                )
            if reqs is not None and len(reqs) > 0:
                f.write(f"\tsys.path.insert(1, '{self.HOME_DIR}/')\n\n")

        f.write(
            "\timport runhouse\n"
            f"\tfrom {handler_name} import {self.handler_function_name}\n"
            "\treturn {\n"
            "\t\t'status_code': 200,\n"
            f"\t\t'body': {self.handler_function_name}(**event)\n"
            "\t}"
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
        lambda_config = self.lambda_client.update_function_configuration(
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

            lambda_config = self.lambda_client.update_function_code(
                FunctionName=self.name, ZipFile=zipped_code
            )

        logger.info(f'{lambda_config["FunctionName"]} was updated successfully.')

        return lambda_config

    def _deploy_lambda_to_aws_helper(self, role_res, zipped_code, env_vars):
        """Deploying new lambda to AWS helping function"""
        try:
            lambda_config = self.lambda_client.create_function(
                FunctionName=self.name,
                Runtime=self.runtime,
                Role=role_res["Role"]["Arn"],
                Handler=f"rh_handler_{self.name}.lambda_handler",
                Code={"ZipFile": zipped_code},
                Timeout=self.timeout,
                MemorySize=self.memory_size,
                Environment={"Variables": env_vars},
                EphemeralStorage={"Size": self.tmp_size},  # size of /tmp folder.
            )
            func_name = lambda_config["FunctionName"]
            with self._wait_until_lambda_update_is_finished(func_name):
                lambda_config = self.lambda_client.get_function(FunctionName=func_name)
                return lambda_config
        except self.lambda_client.exceptions.InvalidParameterValueException or botocore.exceptions.ClientError:
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
            role_res = self.iam_client.create_role(
                RoleName=f"{self.name}_Role",
                AssumeRolePolicyDocument=json.dumps(assume_role_policy_document),
            )
            logger.info(f'{role_res["Role"]["RoleName"]} was created successfully.')

        except self.iam_client.exceptions.EntityAlreadyExistsException:
            role_res = self.iam_client.get_role(RoleName=f"{self.name}_Role")

        self.iam_client.put_role_policy(
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

    def deploy(self):
        return self.to()

    def to(
        self,
        env: Optional[List[str]] = [],
        cloud: str = "aws_lambda",
        force_install: Optional[bool] = False,
    ):
        """
        Set up a function on AWS as a Lambda function.

        See the args of the factory method :func:`aws_lambda_fn` for more information.
        """

        # Checking if the user have a credentials file
        if not Path(CRED_PATH).is_file():
            logger.error("No credentials found")
            raise FileNotFoundError("No credentials found")
        rh_handler_wrapper = self._rh_wrapper()
        self.local_path_to_code.append(rh_handler_wrapper)

        env_vars = (
            self.env.env_vars if isinstance(self.env, Env) else {"HOME": self.HOME_DIR}
        )

        # if function exist - return its aws config. Else, a new one will be created.
        lambda_config = self._lambda_exist(self.name)
        if not lambda_config:
            # creating a new Lambda function, since it's not existing in the AWS account which is configured locally.
            lambda_config = self._create_new_lambda(env_vars)

        # TODO [SB]: in the next phase, enable the user to change the config of the Lambda.
        # updating the configuration with the initial configuration.
        # lambda_config = self._update_lambda_config(env_vars)

        self.aws_lambda_config = lambda_config
        Path(rh_handler_wrapper).absolute().unlink()
        return self

    # --------------------------------------
    # Lambda Function call methods
    # --------------------------------------

    def __call__(self, *args, **kwargs) -> Any:
        """Call (invoke) the Lambdas function

        Args:
             *args: Optional args for the Function
             **kwargs: Optional kwargs for the Function

        Returns:
            The Function's return value
        """
        return self._invoke(*args, **kwargs)

    def _get_log_group_names(self):
        lambdas_log_groups = self.logs_client.describe_log_groups(
            logGroupNamePrefix=LOG_GROUP_PREFIX
        )["logGroups"]
        lambdas_log_groups = [group["logGroupName"] for group in lambdas_log_groups]
        return lambdas_log_groups

    @contextlib.contextmanager
    def _wait_till_log_group_is_created(
        self, curr_lambda_log_group, lambdas_log_groups
    ):
        time_passed = 0
        while curr_lambda_log_group not in lambdas_log_groups:
            if time_passed > self.MAX_WAIT_TIME:
                raise TimeoutError(
                    f"The log group called {curr_lambda_log_group} is being deployed in AWS cloudwatch for too long. "
                    + " Please check the resource in AWS console, delete relevant resource(s) "
                    + "if necessary, and re-run your Runhouse code."
                )
            time.sleep(1)
            lambdas_log_groups = self._get_log_group_names()
            time_passed += 1
        yield True

    def _invoke(self, *args, **kwargs) -> Any:
        lambdas_log_groups = self._get_log_group_names()
        curr_lambda_log_group = f"{LOG_GROUP_PREFIX}{self.name}"
        invoke_for_the_first_time = curr_lambda_log_group not in lambdas_log_groups

        payload_invoke = {}

        if len(args) > 0 and self.args_names is not None:
            payload_invoke = {self.args_names[i]: args[i] for i in range(len(args))}
        invoke_res = self.lambda_client.invoke(
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
                lambdas_log_groups = self._get_log_group_names()
                with self._wait_till_log_group_is_created(
                    curr_lambda_log_group, lambdas_log_groups
                ):
                    self.logs_client.put_retention_policy(
                        logGroupName=curr_lambda_log_group,
                        retentionInDays=self.retention_time,
                    )

            log_lines = "Function Logs are:\n" + base64.b64decode(
                invoke_res["LogResult"]
            ).decode("utf-8")

            print(log_lines)
            return_value = json.loads(return_value)["body"]
            return return_value

    def map(self, *args, **kwargs):
        """Map a function over a list of arguments.

        Example:
            >>> # The my_lambda_handler.py file
            >>> def my_summer(arg1, arg2, arg3):
            >>>     return arg1 + arg2 + arg3
            >>>
            >>> # your 'main' python file, where you are using runhouse
            >>> summer_lambda = rh.aws_lambda_fn(
            >>>                     fn=my_summer,
            >>>                     runtime = 'python3.9',
            >>>                     name="my_func")
            >>> output = summer_lambda.map([1, 2], [1, 4], [2, 3])  # output = [4, 9]
        """

        return [self._invoke(*args, **kwargs) for args in zip(*args)]

    def starmap(self, args_lists, **kwargs):
        """Like :func:`map` except that the elements of the iterable are expected to be iterables
        that are unpacked as arguments. An iterable of [(1,2), (3, 4)] results in [func(1,2), func(3,4)].

        Example:
            >>> arg_list = [(1,2, 3), (3, 4, 5)]
            >>> # invokes the Lambda function twice, once with args (1, 2, 3) and once with args (3, 4, 5)
            >>> output = summer_lambda.starmap(arg_list) # output = [6, 12]
        """

        return [self._invoke(*args, **kwargs) for args in args_lists]

    # --------------------------------------
    # Lambda Function delete methods
    # --------------------------------------
    def teardown(self):
        """Deletes a Lambda function instance from AWS. All relevant AWS resources
        (role, log group) are deleted as well.

        Example:
            >>> def multiply(a, b):
            >>>     return a * b
            >>> multiply_lambda = rh.aws_lambda_fn(fn=multiply, name="lambdas_mult_func")
            >>> mult_res = multiply_lambda(4, 5)  # returns 20.
            >>> multiply_lambda.teardown()  # returns true if succeeded, raises an exception otherwise.

        """
        try:
            lambda_name = self.name
            # delete from aws console
            if self._lambda_exist(lambda_name):
                policy_name = f"{lambda_name}_Policy"
                role_name = f"{lambda_name}_Role"
                log_group_name = f"/aws/lambda/{lambda_name}"

                # TODO: do we want to delete role and log group?
                #  maybe we want to reuse the role or save logs even if the lambda was deleted.
                self.iam_client.delete_role_policy(
                    RoleName=role_name, PolicyName=policy_name
                )

                self.iam_client.delete_role(RoleName=role_name)

                log_groups = self._get_log_group_names()
                if len(log_groups) > 0 and log_group_name in log_groups:
                    self.logs_client.delete_log_group(logGroupName=log_group_name)

                self.lambda_client.delete_function(FunctionName=lambda_name)

            return True
        except botocore.exceptions.ClientError as aws_exception:
            raise RuntimeError(
                f"Could nor delete an AWS resource, got {aws_exception.response['Error']['Message']}"
            )

    def config(self, condensed=True):
        config = super().config(condensed)
        config.update(
            {
                "paths_to_code": self.local_path_to_code,
                "handler_function_name": self.handler_function_name,
                "runtime": self.runtime,
                "timeout": self.timeout,
                "memory_size": self.memory_size,
                "tmp_size": self.tmp_size,
                "args_names": self.args_names,
                "function_arn": self.aws_lambda_config["Configuration"]["FunctionArn"],
            }
        )
        return config

    # --------------------------------------
    # Properties setup
    # --------------------------------------

    @property
    def handler_path(self):
        return self.local_path_to_code[0]
