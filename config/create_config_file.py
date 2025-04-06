import os
from utils.paths import*

def create_config_file():
    config_file_path = f"{dataset_path}/config.yaml"
    config_file_contents = f"""path: {dataset_path}
"""

    with open(config_file_path, 'w') as f:
        f.write(config_file_contents)

    print(f"Config file has been written successfully in: {config_file_path}")

    os.environ['WANDB_DISABLED'] = 'true'

    if not os.path.isfile(config_file_path):
        raise FileNotFoundError(f'Config yaml file not found in: {config_file_path}.')

    return config_file_path
