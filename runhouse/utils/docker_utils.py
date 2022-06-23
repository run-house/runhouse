import os
import docker
import typer
import random
import string
from runhouse.utils.utils import ERROR_FLAG


def create_or_update_docker_ignore(name_dir):
    """Create dockerignore to ignore the runhouse dir"""
    # Ignore the virtual env + the readme
    text = f"""rh/\n**/venv\nREADME*"""
    path_to_docker_ignore_file = os.path.join(name_dir, '.dockerignore')
    with open(path_to_docker_ignore_file, 'w') as f:
        f.write(text)


def create_dockerfile(name_dir):
    # TODO make this cleaner
    text = f"""FROM {os.getenv('DOCKER_PYTHON_VERSION')}\nCOPY requirements.txt /runhouse/requirements.txt\nWORKDIR /runhouse\nRUN pip install -r requirements.txt\nCOPY . .\nARG PYTHONPATH=":/runhouse"\nENV PYTHONPATH=$PYTHONPATH\nCMD ["/bin/bash"]"""
    path_to_docker_file = os.path.join(name_dir, 'Dockerfile')
    with open(path_to_docker_file, 'w') as f:
        f.write(text)

    return path_to_docker_file


def build_image(dockerfile, docker_client, name, tag_name, path_to_parent_dir, hardware):
    """if no image object has been provided we have some work to do"""
    # Need to build the image based on dockerfile provided, or if that isn't provided first build the dockerfile
    try:
        # build it into the user's local docker image store
        resp = docker_client.images.build(path=path_to_parent_dir, dockerfile=dockerfile, tag=tag_name,
                                          labels={'hardware': hardware})
        image_obj = resp[0]
        typer.echo(f"Successfully built image for {name}")
        return image_obj

    except Exception as e:
        typer.echo(f'{ERROR_FLAG} Failed to build image: {e}')
        raise typer.Exit(code=1)


def bring_image_from_local_docker_client(docker_client, image_id):
    try:
        image = docker_client.images.get(image_id)
    except docker.errors.ImageNotFound:
        # if the image doesn't exist
        image = None
    except docker.errors.APIError:
        typer.echo(f'{ERROR_FLAG} Unable to retrieve image')
        raise typer.Exit(code=1)
    return image


def get_path_to_dockerfile(path_to_parent_dir, config_kwargs, ctx):
    # default is in the root directory
    default_path_to_dockerfile = os.path.join(path_to_parent_dir, "Dockerfile")

    # if the dockerfile is specified in the cli options or the config then take that value instead
    dockerfile_path_from_user = ctx.obj.dockerfile or config_kwargs.get('dockerfile')
    return default_path_to_dockerfile or dockerfile_path_from_user


def launch_local_docker_client():
    try:
        docker_client = docker.from_env()
        return docker_client
    except docker.errors.DockerException:
        typer.echo(f'{ERROR_FLAG} Docker client error')
        raise typer.Exit(code=1)


def dockerfile_has_changed(dockerfile_time_added: float, path_to_dockerfile: str):
    """If the dockerfile has been updated since it was first created (as indicated in the config file)"""
    time_modified = os.path.getctime(path_to_dockerfile)
    return time_modified - dockerfile_time_added > 100


def generate_image_id(length=12):
    """Create random id to assign to the image"""
    # choose from all lowercase letter
    letters = string.ascii_lowercase
    result_str = ''.join(random.choice(letters) for i in range(length))
    return result_str


def image_tag_name(name, image_id):
    return f'{name}-{image_id}'


def full_ecr_tag_name(tag_name):
    return f'{os.getenv("ECR_URI")}:{tag_name}'
