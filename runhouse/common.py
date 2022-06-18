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

    # Options we want to specifically check if changed by the user between runs
    OPTIONS_TO_CHECK = {'hardware', 'dockerfile', 'file', 'image', 'shell', 'path'}

    @property
    def args_to_check(self):
        return {k: v for k, v in vars(self).items() if k in self.OPTIONS_TO_CHECK}

    @property
    def user_provided_args(self):
        """Check which CLI options the user explicitly provided"""
        return {k: v for k, v in self.args_to_check.items() if v is not None}
