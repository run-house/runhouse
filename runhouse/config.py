import os
from configparser import ConfigParser
import typer

from runhouse.utils.docker_utils import dockerfile_has_changed
from runhouse.utils.utils import ERROR_FLAG, current_time
from runhouse.utils.validation import valid_filepath


class Config:
    CONFIG_FILE = 'config.ini'
    MAIN_CONF_HEADER = 'main'
    DOCKER_CONF_HEADER = 'docker'

    def __init__(self):
        self.config = ConfigParser(strict=False, allow_no_value=True)

    def write_config(self, config_path):
        try:
            with open(config_path, 'w') as f:
                self.config.write(f)
        except:
            typer.echo(f'{ERROR_FLAG} Unable to save self.config file')
            raise typer.Exit(code=1)

    def create_or_update_config_file(self, directory, **kwargs):
        """Add or update existing file"""
        config_path = os.path.join(directory, self.CONFIG_FILE)
        rename = kwargs.get('rename')

        if rename:
            if not valid_filepath(config_path):
                # If we are trying to rename an existing self.config make sure it still exists
                typer.echo(f'{ERROR_FLAG} Invalid path to self.config file')
                raise typer.Exit(code=1)

            # All we care about here is the actual "name" field defined in the config
            new_kwargs = self.read_config_file(config_path)
            new_kwargs['name'] = kwargs.get('name')
            new_kwargs['rename'] = False
            self.create_or_update_config_file(directory, **new_kwargs)
            return

        self.config.read(self.CONFIG_FILE)

        if not self.config.has_section(self.MAIN_CONF_HEADER):
            self.config.add_section(self.MAIN_CONF_HEADER)

        if not self.config.has_section(self.DOCKER_CONF_HEADER):
            self.config.add_section(self.DOCKER_CONF_HEADER)

        dockerfile = kwargs.get('dockerfile')
        rebuild = kwargs.get('rebuild')

        self.config.set(self.MAIN_CONF_HEADER, 'name', kwargs.get('name'))
        self.config.set(self.MAIN_CONF_HEADER, 'hardware', kwargs.get('hardware', os.getenv('DEFAULT_HARDWARE')))
        self.config.set(self.MAIN_CONF_HEADER, 'path', str(kwargs.get('path')))
        self.config.set(self.MAIN_CONF_HEADER, 'file', kwargs.get('file'))
        self.config.set(self.MAIN_CONF_HEADER, 'last_run', str(current_time()))

        self.config.set(self.DOCKER_CONF_HEADER, 'dockerfile', dockerfile)
        self.config.set(self.DOCKER_CONF_HEADER, 'image_id', kwargs.get('image_id'))
        self.config.set(self.DOCKER_CONF_HEADER, 'image_path', kwargs.get('image_path'))

        dockerfile_time_added = kwargs.get('config_kwargs', {}).get('dockerfile_time_added')
        if rebuild or not valid_filepath(dockerfile) or dockerfile_has_changed(float(dockerfile_time_added),
                                                                               path_to_dockerfile=dockerfile):
            # Update the time added if we are doing a rebuild or the dockerfile doesn't exist / has been updated
            self.config.set(self.DOCKER_CONF_HEADER, 'dockerfile_time_added', str(current_time()))

        self.write_config(config_path)

    def read_config_file(self, config_path):
        """Parse existing config file"""
        self.config.read(config_path)

        # read values from file
        dockerfile = self.config.get(self.DOCKER_CONF_HEADER, 'dockerfile')
        image_id = self.config.get(self.DOCKER_CONF_HEADER, 'image_id')
        image_path = self.config.get(self.DOCKER_CONF_HEADER, 'image_path')
        dockerfile_timestamp = self.config.get(self.DOCKER_CONF_HEADER, 'dockerfile_time_added')

        name = self.config.get(self.MAIN_CONF_HEADER, 'name')
        hardware = self.config.get(self.MAIN_CONF_HEADER, 'hardware')
        path = self.config.get(self.MAIN_CONF_HEADER, 'path')
        file = self.config.get(self.MAIN_CONF_HEADER, 'file')

        return {'dockerfile': dockerfile, 'image_id': image_id, 'image_path': image_path, 'name': name,
                'hardware': hardware, 'path': path, 'file': file, 'dockerfile_time_added': dockerfile_timestamp}

    def bring_config_kwargs(self, config_path, name, file):
        if not valid_filepath(config_path):
            # If we don't have a config for this name yet define the initial default values
            return {'name': name, 'hardware': os.getenv('DEFAULT_HARDWARE'), 'file': file}

        # take from the config that already exists
        return self.read_config_file(config_path)

