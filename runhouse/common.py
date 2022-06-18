from dataclasses import dataclass


@dataclass
class Common:
    """Context manager for all the possible CLI options the user can provide"""
    name: str
    hardware: str
    dockerfile: str
    file: str
    image: str
    shell: bool
    path: str
    anon: bool
    rename: str

    @property
    def args_to_check(self):
        return {'hardware': self.hardware, 'dockerfile': self.dockerfile, 'file': self.file, 'image': self.image,
                'shell': self.shell, 'path': self.path}

    @property
    def user_provided_args(self):
        """Check which CLI options the user explicitly provided"""
        return {k: v for k, v in self.args_to_check.items() if v is not None}
