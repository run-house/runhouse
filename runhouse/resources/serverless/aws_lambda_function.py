import base64
import inspect
import json
import logging
import os
import shutil
import time
import zipfile
from pathlib import Path
from typing import Any, Callable, List, Optional

import boto3

from runhouse.globals import rns_client

from runhouse.resources.function import Function

logger = logging.getLogger(__name__)
CRED_PATH_MAC = f"{Path.home()}/.aws/credentials"
CRED_PATH_WIN = f"{Path.home()}\.aws\credentials"
DEFAULT_REGION = "us-east-1"

if Path(CRED_PATH_MAC).is_file() or Path(CRED_PATH_WIN).is_file():
    LAMBDA_CLIENT = boto3.client("lambda")
else:
    LAMBDA_CLIENT = boto3.client("lambda", region_name=DEFAULT_REGION)

SUPPORTED_RUNTIMES = [
    "python3.7",
    "python3.8",
    "python3.9",
    "python3.10",
    "python 3.11",
]


class AWSLambdaFunction(Function):
    RESOURCE_TYPE = "aws_lambda"
    DEFAULT_ROLE_POLICIES = [
        "cloudwatch:*",
        "lambda:Invoke",
        "lambda:InvokeAsync",
        "lambda:InvokeFunction",
        "lambda:PublishLayerVersion",
        "lambda:PublishVersion",
        "lambda:GetLayerVersion",
        "lambda:GetLayerVersionPolicy",
        "lambda:ListLayers",
        "lambda:ListLayerVersions",
        "logs:*",
        "s3:DeleteObject",
        "s3:GetObject",
        "s3:ListBucket",
        "s3:PutObject",
    ]
    GEN_ERROR = "could not create or update the AWS Lambda."
    FAIL_CODE = 1
    DEFAULT_REGION = "us-east-1"
    DEFAULT_PY_VERSION = "python3.9"
    EMPTY_ZIP = -1

    def __init__(
        self,
        paths_to_code: Optional[list[str]] = None,
        handler_function_name: Optional[str] = None,
        runtime: Optional[str] = None,
        args_names: Optional[list[str]] = None,
        fn: Optional[Callable] = None,
        fn_pointers: Optional[tuple] = None,
        name: Optional[str] = None,
        env: Optional[list[str] or str] = None,
        env_vars: Optional[dict] = None,
        dryrun: bool = False,
        timeout: Optional[int] = 30,  # seconds
        memory_size: Optional[int] = 128,  # MB
        **kwargs,  # We have this here to ignore extra arguments when calling from from_config
    ):
        """
        Runhouse AWS Lambda object. It is comprised of the entry point, configuration,
        and dependencies necessary to run the service.

        .. note::
                To create an AWS lambda resource, please use the factory method :func:`aws_lambda_function`.
        """
        if isinstance(fn, Callable):
            handler_function_name = fn.__name__
            fn_pointers = Function._extract_pointers(fn, reqs=env or [])
            paths_to_code = self._paths_to_code_from_fn_pointers(fn_pointers)
            args_names = [
                param.name for param in inspect.signature(fn).parameters.values()
            ]
        elif fn_pointers:
            handler_function_name = fn_pointers[2]
            paths_to_code = self._paths_to_code_from_fn_pointers(fn_pointers)
        else:
            fn_pointers = None
            if handler_function_name is None or paths_to_code is None:
                raise ValueError(
                    "`handler_function_name` and `paths_to_code` must be provided if `fn` is not provided"
                )

        if name is None:
            name = handler_function_name.replace(".", "_")

        super().__init__(
            name=name,
            dryrun=dryrun,
            system="AWS_lambda",
            env=None,
            fn_pointers=fn_pointers,
            **kwargs,
        )

        self.local_path_to_code = paths_to_code
        self.handler_function_name = handler_function_name
        self.args_names = args_names
        self.runtime = runtime or self.DEFAULT_PY_VERSION
        self.env_vars = env_vars

        if timeout > 900:
            timeout = 900
        if timeout < 3:
            timeout = 3
        self.timeout = timeout

        if memory_size < 128:
            memory_size = 128
        if memory_size > 10240:
            memory_size = 10240
        self.memory_size = memory_size

        if env:
            if type(env) is list:
                self.reqs = env
            else:
                self.reqs = self._reqs_to_list(env)
            (
                self.layer,
                self.layer_version,
                self.np_layer,
                self.np_layer_version,
            ) = self._create_layer()  # returns layers ARNs and versions in AWS.
        else:
            (
                self.reqs,
                self.layer,
                self.layer_version,
                self.np_layer,
                self.np_layer_version,
            ) = (None, None, None, None, None)
        self.aws_lambda_config = (
            None  # Lambda config and role arn from aws will be saved here
        )

    # ----------------- Constructor helper methods -----------------

    @classmethod
    def from_config(cls, config: dict, dryrun: bool = False):
        """Create an AWS Lambda object from a config dictionary."""

        if "resource_subtype" in config.keys():
            config.pop("resource_subtype", None)
        if "system" in config.keys():
            config.pop("system", None)

        return AWSLambdaFunction(**config, dryrun=dryrun).to()

    @classmethod
    def from_name(cls, name, dryrun=False, alt_options=None):
        if cls._lambda_exist(cls, name):
            func = super().from_name(name=name)
            return func.to()
        else:
            logger.error(
                f"Could not find a Lambda called {name}. Please provide a name of an existing Lambda, "
                + "or paths_to_code, handler_function_name, runtime and args_names (and a name if you"
                " wish), in order to create a new lambda."
            )
            return "LambdaNotFoundInAWS"

    # ----------------- Private helping methods -----------------

    @classmethod
    def _check_for_child_configs(cls, config):
        """Overload this method of the function class."""
        return config

    @staticmethod
    def _paths_to_code_from_fn_pointers(fn_pointers):
        root_dir = rns_client.locate_working_dir()
        file_path = fn_pointers[1].replace(".", "/") + ".py"
        paths_to_code = [os.path.join(root_dir, file_path)]
        return paths_to_code

    def _reqs_to_list(self, env):
        """Converting requirements from requirements.txt to a list"""

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
            func["FunctionName"] for func in LAMBDA_CLIENT.list_functions()["Functions"]
        ]
        return name in func_names

    def _wait_until_update_is_finished(self, name):
        """Verifies that a running update of the function (in AWS) is finished (so the next one could be executed)"""
        response = LAMBDA_CLIENT.get_function(FunctionName=name)
        state = response["Configuration"]["State"] == "Active"
        last_update_status = (
            response["Configuration"]["LastUpdateStatus"] != "InProgress"
        )

        while True:
            if state and last_update_status:
                break
            else:
                time.sleep(1)
                response = LAMBDA_CLIENT.get_function(FunctionName=name)
                state = response["Configuration"]["State"] == "Active"
                last_update_status = (
                    response["Configuration"]["LastUpdateStatus"] != "InProgress"
                )
        return True

    def _rh_wrapper(self):
        """Creates a runhouse wrapper to the handler function"""
        handler_path = self.local_path_to_code[0]
        wrapper_path = str(Path(handler_path).parent / f"rh_handler_{self.name}.py")
        handler_name = Path(handler_path).stem

        f = open(wrapper_path, "w")
        f.write(f"from {handler_name} import {self.handler_function_name}\n")
        f = open(wrapper_path, "w")
        f.write(f"from {handler_name} import {self.handler_function_name}\n")
        if self.reqs:
            for req in self.reqs:
                f.write(f"import {req}\n")
            f.write("\n\n")
        f.write(
            "def lambda_handler(event, context):\n"
            f"\treturn {self.handler_function_name}(**event)"
        )
        f.close()
        return wrapper_path

    def _supported_python_libs(self):
        """ " Returns a list of the supported python libs by the AWS Lambda resource"""
        # TODO [SB]: think what is the better implementation: via website, or AWS lambda. for now, hard-coded.
        # url = "https://www.feitsui.com/en/article/2"
        # page = urlopen(url)
        # html_bytes = page.read()
        # html = html_bytes.decode("utf-8")
        # python_version = self.runtime[0].upper()+self.runtime[1:-3]+" "+self.runtime[-3:]
        # python_prev_version = python_version[:-1]+str(int(python_version[-1])-1)
        # # TODO [SB]: no indexes, last python version
        # start_index = html.index(python_version)
        # end_index = html.index(python_prev_version)
        # version_text = html[start_index:end_index]
        # version_text = version_text.split("\n")
        # version_text = [v.strip() for v in version_text]
        # version_text = [v for v in version_text if "<td>" in v]
        # version_text = [v[4:-5] for v in version_text]
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

    def _download_packages_s3(self, dest_path):
        """Getting required layer resources from a s3 bucket."""
        bucket_name = "runhouse-lambda-resources"
        remote_path = f"layer_helpers/{self.runtime}"
        try:

            os.system(
                f"aws s3 cp s3://{bucket_name}/{remote_path} {dest_path} --recursive"
            )
        except Exception:
            dir_name = str(Path(__file__).parent / "all_reqs")
            shutil.rmtree(dir_name)
            raise Exception(
                "Try installing aws on your local machine (i.e brew install awscli) and rerun."
            )

    def _create_layer_zip(self):
        """Creates a zip of all required python libs, that will be sent to the Lambda as a layer"""
        supported_libs = self._supported_python_libs()
        reqs = [req for req in self.reqs if req not in supported_libs]
        dir_name = str(Path(__file__).parent / "all_reqs")
        all_req_dir = Path(__file__).parent / "all_reqs" / "python"
        Path(all_req_dir).mkdir(parents=True, exist_ok=True)

        if "numpy" in reqs:
            reqs.remove("numpy")
        if "pandas" in reqs:
            reqs.remove("pandas")
        if len(reqs) == 0:
            self.layer, self.layer_version = None, None
            shutil.rmtree(dir_name)
            return self.EMPTY_ZIP
        for req in reqs:
            folder_req = str(Path(__import__(req).__file__).parent)
            folder_req = folder_req.replace(f"/{req}", "")
            sub_reqs = [
                str(item) for item in Path(folder_req).iterdir() if req in str(item)
            ]
            for r in sub_reqs:
                shutil.copytree(folder_req + "/" + r, str(all_req_dir) + "/" + r)

        shutil.make_archive(dir_name, "zip", dir_name)
        shutil.rmtree(dir_name)
        return dir_name + ".zip"

    def _create_layer(self):
        """Creates a layer, which contains required python libs.
        If needed, pandas and numpy layer is also created and returned
        Returns layers' ARNs and versions"""

        layer_name, layer_arn, layer_version = self.name + "_layer", None, None
        zip_file_name = self._create_layer_zip()

        # Creating layer of python libs which are not np, pd and are not supported by AWS Lambda by default.
        if zip_file_name != self.EMPTY_ZIP:
            reqs_no_np = [req for req in self.reqs if req != "numpy" or req != "pandas"]
            description = f"This layer contains the following python libraries: {', '.join(reqs_no_np)}"
            with open(zip_file_name, "rb") as f:
                layer_zf = f.read()
            layer = LAMBDA_CLIENT.publish_layer_version(
                LayerName=layer_name,
                Description=description,
                Content={"ZipFile": layer_zf},
                CompatibleRuntimes=[self.runtime],
            )
            layer_arn, layer_version = layer["LayerVersionArn"], int(
                layer["LayerVersionArn"].split(":")[-1]
            )
            Path(zip_file_name).unlink()

        # Creating and getting the np and pd layer, suitable to the runtime provided during init.
        list_layers = LAMBDA_CLIENT.list_layers(
            CompatibleRuntime=f"{self.runtime}", CompatibleArchitecture="x86_64"
        )

        layer_names = [layer["LayerName"] for layer in list_layers["Layers"]]

        pd_np_layer_name = f"numpy_pandas_{self.runtime}"
        pd_np_layer_name = pd_np_layer_name.replace(".", "_")

        # create the layer if not existing in AWS.
        if pd_np_layer_name not in layer_names:
            np_layer = LAMBDA_CLIENT.publish_layer_version(
                LayerName=pd_np_layer_name,
                Description=f"This layer contains numpy and pandas suitable for {self.runtime}",
                Content={
                    "S3Bucket": "runhouse-lambda-resources",
                    "S3Key": f"layer_helpers/{self.runtime}/python.zip",
                },
                CompatibleRuntimes=[self.runtime],
            )
            pd_np_layer_arn = np_layer["LayerVersionArn"]
            pd_np_layer_version = np_layer["Version"]

        # get the layer version and ARN if existing in AWS.
        else:
            pd_np_layer_arn = [
                layer["LatestMatchingVersion"]["LayerVersionArn"]
                for layer in list_layers["Layers"]
                if layer["LayerName" == pd_np_layer_name]
            ][0]
            pd_np_layer_version = [
                layer["LayerArn"]
                for layer in list_layers["Layers"]
                if layer["LayerName"] == pd_np_layer_name
            ]
            pd_np_layer_version = pd_np_layer_version[0].split(":")[-1]

        return layer_arn, layer_version, pd_np_layer_arn, pd_np_layer_version

    def _update_lambda_config(self, env_vars):
        """Updates existing Lambda in AWS (config) that was provided in the init."""
        time.sleep(4)
        logger.info(f"Updating a Lambda called {self.name}")
        layers = []
        if self.layer:
            layers.append(self.layer)
        if self.np_layer:
            layers.append(self.np_layer)
        if len(layers) > 0:
            lambda_config = LAMBDA_CLIENT.update_function_configuration(
                FunctionName=self.name,
                Runtime=self.runtime,
                Timeout=self.timeout,
                MemorySize=self.memory_size,
                Layers=layers,
                Environment={"Variables": env_vars},
            )
        else:
            lambda_config = LAMBDA_CLIENT.update_function_configuration(
                FunctionName=self.name,
                Runtime=self.runtime,
                Timeout=self.timeout,
                MemorySize=self.memory_size,
                Environment={"Variables": env_vars},
            )

        # TODO: enable for other to update the Lambda code.
        # wait for the config update process to finish, and then update the code (Lambda logic).
        if self._wait_until_update_is_finished(self.name):
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

            lambda_config = LAMBDA_CLIENT.update_function_code(
                FunctionName=self.name, ZipFile=zipped_code
            )

        logger.info(f'{lambda_config["FunctionName"]} was updated successfully.')

        return lambda_config

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
        # TODO: enable the user to update the default policy

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
            time.sleep(5)
            logger.info(f'{role_res["Role"]["RoleName"]} was created successfully.')

        except iam_client.exceptions.EntityAlreadyExistsException:
            role_res = iam_client.get_role(RoleName=f"{self.name}_Role")

            iam_client.put_role_policy(
                RoleName=role_res["Role"]["RoleName"],
                PolicyName=f"{self.name}_Policy",
                PolicyDocument=json.dumps(role_policy),
            )

            time.sleep(4)  # letting the role be updated in AWS

        layers = []
        if self.layer:
            LAMBDA_CLIENT.add_layer_version_permission(
                LayerName=self.name + "_layer",
                VersionNumber=self.layer_version,
                StatementId=role_res["Role"]["RoleId"],
                Action="lambda:GetLayerVersion",
                Principal="*",
            )
            time.sleep(5)  # letting the role be updated in AWS

            layers.append(self.layer)

        if self.np_layer:
            layers.append(self.np_layer)
            layer_name = self.np_layer.split(":")[-2]
            LAMBDA_CLIENT.add_layer_version_permission(
                LayerName=layer_name,
                VersionNumber=self.np_layer_version,
                StatementId=role_res["Role"]["RoleId"],
                Action="lambda:GetLayerVersion",
                Principal="*",
            )
            time.sleep(5)  # letting the role be updated in AWS

        if len(layers) > 0:

            lambda_config = LAMBDA_CLIENT.create_function(
                FunctionName=self.name,
                Runtime=self.runtime,
                Role=role_res["Role"]["Arn"],
                Handler=f"rh_handler_{self.name}.lambda_handler",
                Code={"ZipFile": zipped_code},
                Timeout=self.timeout,
                MemorySize=self.memory_size,
                Layers=layers,
                Environment={"Variables": env_vars},
            )
        else:
            lambda_config = LAMBDA_CLIENT.create_function(
                FunctionName=self.name,
                Runtime=self.runtime,
                Role=role_res["Role"]["Arn"],
                Handler=f"rh_handler_{self.name}.lambda_handler",
                Code={"ZipFile": zipped_code},
                Timeout=self.timeout,
                MemorySize=self.memory_size,
                Environment={"Variables": env_vars},
            )
        time.sleep(4)
        logger.info(f'{lambda_config["FunctionName"]} was created successfully.')
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
        if not (Path(CRED_PATH_MAC).is_file() or Path(CRED_PATH_WIN).is_file()):
            logger.error(f"No credentials found, {self.GEN_ERROR}")
            raise Exception("No credentials found")

        rh_handler_wrapper = self._rh_wrapper()
        self.local_path_to_code.append(rh_handler_wrapper)
        # self.local_path_to_code.pop(0)

        env_vars = self.env_vars if self.env_vars else {}

        # if function exist - will update it. Else, a new one will be created.
        if self._lambda_exist(self.name):
            # updating the configuration with the initial configuration.
            # TODO: enable the user to change the config of the Lambda.
            lambda_config = self._update_lambda_config(env_vars)

        else:
            # creating a new Lambda function, since it's not existing in the AWS account which is configured locally.
            lambda_config = self._create_new_lambda(env_vars)

        self.aws_lambda_config = lambda_config
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

    def _invoke(self, *args, **kwargs) -> Any:
        payload_invoke = {}
        if len(args) > 0:
            payload_invoke = {self.args_names[i]: args[i] for i in range(len(args))}
        invoke_res = LAMBDA_CLIENT.invoke(
            FunctionName=self.name,
            Payload=json.dumps({**payload_invoke, **kwargs}),
            LogType="Tail",
        )
        return_value = invoke_res["Payload"].read().decode("utf-8")
        try:
            logger.error(invoke_res["FunctionError"])
            raise RuntimeError(
                f"Failed to run {self.name}: {invoke_res['FunctionError']}"
            )
        except KeyError:
            log_lines = "Function Logs are:\n" + base64.b64decode(
                invoke_res["LogResult"]
            ).decode("utf-8")
            for line in log_lines:
                logger.info(line)

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
                "reqs": self.reqs,
                "layer": self.layer,
                "env_vars": self.env_vars,
            }
        )
        return config


def aws_lambda_function(
    fn: Callable = None,
    paths_to_code: list[str] = None,
    handler_function_name: str = None,
    runtime: str = None,
    args_names: list[str] = None,
    name: Optional[str] = None,
    env: Optional[list[str] or str] = None,
    env_vars: Optional[dict] = None,
    timeout: Optional[int] = 30,
    memory_size: Optional[int] = 128,
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
        env (Optional[List[str] or str]): Specifies the requirements (python libraries), which will be used by the
            Lambda.Can be a list or path to the requirements.txt file. All provided libraries are required to be
            installed locally.
        env_vars: Optional[dict]: Dictionary of enviroumnt varible name (key) and its value. They will be
            deployed as enviroumnt varibles, which are a port of the Lambda configuration.
        timeout: Optional[int]: The maximum amount of time (in secods) during which the Lambda will run in AWS
            without timing-out. (Default: ``30``, Min: ``3``, Max: ``900``)
        memory_size: Optional[int], The amount of memeory, im MB, that will be aloocatied to the lambda.
             (Default: ``128``, Min: ``128``, Max: ``10240``)
        dryrun (bool): Whether to create the Function if it doesn't exist, or load the Function object as a dryrun.
            (Default: ``False``). Is not used by Lambda, but is a port of Function constructor signatuere.

    Returns:
        AWSLambdaFunction: The resulting AWS Lambda Function object.

        .. note::
            1. Some older python versions are not suporrted by the latest numpy and pandas versions.
            When creating a numpy or pandas layer, their version will be according to the Lambda's python version
            (aka Lambda's runtime).\n
            2. When creating the function for the first time (and not reloading it), the following arguments are
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
    if not any([name, paths_to_code, handler_function_name, runtime, fn, args_names]):
        logger.error(
            "Runhouse can't create a Lambda function. "
            + "Please provide lambda's name and/or paths_to_code, handler_function_name, runtime and args_names"
        )
        raise Exception("NoEnoughArgsProvided")

    if name and not any(
        [paths_to_code, handler_function_name, runtime, fn, args_names]
    ):
        # Try reloading existing function
        return AWSLambdaFunction.from_name(name=name)

    # ------- arguments validation -------
    if paths_to_code is None or len(paths_to_code) == 0:
        logger.error("Please provide a path to the lambda handler file.")
        raise RuntimeError
    if handler_function_name is None or len(handler_function_name) == 0:
        logger.error(
            "Please provide the name of the function that should be executed by the lambda."
        )
        raise RuntimeError
    if runtime is None or runtime not in SUPPORTED_RUNTIMES:
        logger.error(
            f"Please provide a supported lambda runtime, should be one of the following: {SUPPORTED_RUNTIMES}"
        )
        raise RuntimeError
    if args_names is None:
        logger.error(
            "Please provide the names of the arguments provided to handler function, in the order they are"
            + " passed to the lambda function."
        )
        raise RuntimeError

    # TODO: [SB] in the next phase, maybe add the option to create func from git.

    fn_pointers = (
        Function._extract_pointers(fn, reqs=env or []) if callable(fn) else None
    )

    new_function = AWSLambdaFunction(
        fn=fn,
        fn_pointers=fn_pointers,
        paths_to_code=paths_to_code,
        handler_function_name=handler_function_name,
        runtime=runtime,
        args_names=args_names,
        name=name,
        env=env,
        env_vars=env_vars,
        dryrun=dryrun,
        timeout=timeout,
        memory_size=memory_size,
    ).to()

    return new_function
