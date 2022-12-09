import logging
from pathlib import Path
import sys
import subprocess
from typing import Optional, List

import ray.cloudpickle as pickle
from runhouse.rns.folder import Folder
from runhouse import rh_config

INSTALL_METHODS = {'local', 'reqs', 'pip', 'conda', 'git', 'gh', 'rh'}


class Package(Folder):
    RESOURCE_TYPE = 'package'

    def __init__(self,
                 name: str = None,
                 url: str = None,
                 fs: str = Folder.DEFAULT_FS,
                 install_method: str = None,
                 local_mount: bool = False,
                 data_config: dict = {},
                 save_to: Optional[List[str]] = None,
                 dryrun: bool = False,
                 **kwargs  # We have this here to ignore extra arguments when calling from from_config
                 ):
        super().__init__(name=name,
                         url=url,
                         fs=fs,
                         local_mount=local_mount,
                         data_config=data_config,
                         save_to=save_to,
                         dryrun=dryrun,
                         )
        self.install_method = install_method

    @property
    def config_for_rns(self):
        # If the package is just a simple Package.from_string string, no
        # need to store it in rns, just give back the string.
        if self.install_method in ['pip', 'conda', 'git']:
            return f'{self.install_method}:{self.name}'
        config = super().config_for_rns
        config['install_method'] = self.install_method
        return config

    def install(self):
        logging.info(f'Installing package {self.name} with method {self.install_method}.')
        if self.install_method in ['local', 'reqs']:
            local_path = self.local_path
            if not local_path:
                local_path = '~/' + self.name
            elif not self.is_local():
                local_path = self.mount(url=f'~/{Path(self.url).stem}')

            if self.install_method == 'reqs':
                if (Path(local_path) / 'requirements.txt').exists():
                    logging.info(f'Attempting to install requirements from {local_path}/requirements.txt')
                    self.pip_install(f'-r {Path(local_path)}/requirements.txt')
                else:
                    logging.info(f'{local_path}/requirements.txt not found, skipping')
                sys.path.append(local_path)
            else:
                if (Path(local_path) / 'setup.py').exists():
                    self.pip_install(f'-e {local_path}')
                elif (Path(local_path) / 'pyproject.toml').exists():
                    self.pip_install(f'{local_path}')
                else:
                    sys.path.append(local_path)
        elif self.install_method == 'git':
            self.pip_install(f'git+{self.url}')
        elif self.install_method == 'gh':
            self.pip_install(f'git+ssh://git@{self.url}')
        elif self.install_method == 'pip':
            self.pip_install(self.name)
        elif self.install_method == 'conda':
            subprocess.run(['conda', 'install', '-y', self.name])
        elif self.install_method == 'unpickle':
            # unpickle the functions and make them importable
            with self.get('functions.pickle') as f:
                sys.modules[self.name] = pickle.load(f)

    @staticmethod
    def pip_install(package_str):
        logging.info(f'Installing: pip install {package_str}')
        subprocess.check_call([sys.executable, "-m", "pip", "install"] + package_str.split(' '))

    @staticmethod
    def from_config(config: dict, dryrun=False):
        return Package(**config, dryrun=dryrun)

    @staticmethod
    def from_string(specifier: str, dryrun=False):
        if specifier.startswith('local:'):
            return Package(url=specifier[6:], install_method='local', dryrun=dryrun)
        elif specifier.startswith('reqs:'):
            return Package(url=specifier[5:], install_method='reqs', dryrun=dryrun)
        elif specifier.startswith('git+') or specifier.startswith('git:'):
            return Package(url=specifier[4:], install_method='git', dryrun=dryrun)
        elif specifier.startswith('gh:'):
            # TODO [DG] test and make this actually work
            return Package(url='github.com/' + specifier[3] + '.git', install_method='git', dryrun=dryrun)
        elif specifier.startswith('pip:'):
            return Package(name=specifier[4:], install_method='pip', dryrun=dryrun)
        elif specifier.startswith('conda:'):
            return Package(name=specifier[6:], install_method='conda', dryrun=dryrun)
        elif specifier.startswith('rh:'):
            # Calling the factory function below!
            return package(name=specifier[4:], dryrun=dryrun)
        else:
            if Path(specifier).exists():
                return Package(url=specifier, install_method='reqs', dryrun=dryrun)
            else:
                return Package(name=specifier, install_method='pip', dryrun=dryrun)


def package(name=None,
            url=None,
            fs='file',
            install_method=None,
            save_to: Optional[List[str]] = None,
            load_from: Optional[List[str]] = None,
            dryrun=False,
            local_mount: bool = False,
            data_config: dict = {},
            ):
    config = rh_config.rns_client.load_config(name, load_from=load_from)
    config['name'] = name or config.get('rns_address', None) or config.get('name')
    config['url'] = url or config.get('url')
    config['fs'] = fs or config.get('fs')
    config['install_method'] = install_method or config.get('install_method')
    config['local_mount'] = local_mount or config.get('local_mount')
    config['data_config'] = data_config or config.get('data_config')
    config['save_to'] = save_to

    new_package = Package.from_config(config, dryrun=dryrun)

    if new_package.name:
        new_package.save()

    return new_package
