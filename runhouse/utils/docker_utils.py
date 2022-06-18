import os
import docker
import typer

from runhouse.utils.utils import ERROR_FLAG, current_time
from runhouse.utils.validation import valid_filepath


def create_dockerfile(path_to_reqs, name_dir):
    # TODO make this cleaner
    text = f"""FROM {os.getenv('DOCKER_PYTHON_VERSION')}\nCOPY {path_to_reqs} 
    /opt/app/requirements.txt\nWORKDIR /opt/app\nRUN pip install -r {path_to_reqs}\nCOPY . .\nCMD [ "echo", "finished building image for {name_dir}" ]"""
    path_to_docker_file = os.path.join(name_dir, 'Dockerfile')
    with open(path_to_docker_file, 'w') as f:
        f.write(text)
    return path_to_docker_file


def build_and_save_image(image, path_to_image, path_to_reqs, dockerfile, docker_client, name, name_dir, hardware):
    success = True
    if not image:
        # if no image url has been provided we have some work to do
        # Need to build the image based on dockerfile provided, or if that isn't provided first build the dockerfile
        if not dockerfile:
            typer.echo('Building Dockerfile')
            dockerfile = create_dockerfile(path_to_reqs, name_dir)

        try:
            docker_client.images.build(path=name_dir, dockerfile=dockerfile, tag=name, labels={'hardware': hardware})
            typer.echo(f'[1/3] Successfully built image for {name}')
        except:
            typer.echo(f'[1/3] {ERROR_FLAG} Failed to build image')
            success = False

    else:
        # TODO this flow should change
        # if image exists then load into compressed format to be shipped off remotely
        save_image_to_tar(image, path_to_image)
        typer.echo(f"[1/3] Successfully loaded image {image.tags} with labels: {image.labels}")

    return success


def bring_image_from_docker_client(docker_client, image_id):
    if image_id is None:
        # We may not have any image at this stage
        return None
    try:
        image = docker_client.images.get(image_id)
    except docker.errors.ImageNotFound:
        # if the image doesn't exist
        image = None
    except docker.errors.APIError:
        typer.echo(f'{ERROR_FLAG} Unable to retrieve image')
        raise typer.Exit(code=1)
    return image


def save_image_to_tar(image, path_to_image):
    if valid_filepath(path_to_image):
        # if already saved by user we're good
        return

    f = open(path_to_image, 'wb')
    for chunk in image.save(chunk_size=2097152, named=False):
        f.write(chunk)
    f.close()


def get_path_to_dockerfile(path_to_parent_dir, config_kwargs, ctx):
    path_to_dockerfile = os.path.join(path_to_parent_dir, "Dockerfile")
    if valid_filepath(path_to_dockerfile):
        return path_to_dockerfile

    # if the dockerfile doesn't yet exist in filesystem bring it from the user CLI option or the config file
    return ctx.obj.dockerfile or config_kwargs.get('dockerfile')


def launch_local_docker_client():
    try:
        docker_client = docker.from_env()
        return docker_client
    except docker.errors.DockerException:
        typer.echo(f'{ERROR_FLAG} Docker client error')
        raise typer.Exit(code=1)


def default_image_name(name):
    return f'{name}_{int(current_time())}'


def dockerfile_has_changed(dockerfile_time_added: float, path_to_dockerfile: str):
    """If the dockerfile has been updated since it was first created (as indicated in the config file)"""
    time_modified = os.path.getctime(path_to_dockerfile)
    return time_modified - dockerfile_time_added > 100
