from pathlib import Path
import yaml

def current_cluster_name():
    if Path('~/.sky/sky_ray.yml').expanduser().exists():
        with open(Path('~/.sky/sky_ray.yml').expanduser()) as f:
            ray_config_data = yaml.safe_load(f)
        return ray_config_data['cluster_name']
    else:
        return None

THIS_CLUSTER = current_cluster_name()