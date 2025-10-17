#最初からtestsのpyを実行するべきだった(機能的には同じ)

import os
import glob
import re

config_dir = "/home/tm/img-science/github/PyTorch-VAE/configs"
original_files = glob.glob(os.path.join(config_dir, "*.yaml"))

# Create a directory for temporary configs if it doesn't exist
tmp_config_dir = os.path.join(config_dir, "tmp")
os.makedirs(tmp_config_dir, exist_ok=True)

for file_path in original_files:
    file_name = os.path.basename(file_path)
    if file_name == "vae.yaml" or file_name.startswith("tmp_"):
        continue

    with open(file_path, 'r') as f:
        content = f.read()

    # Modify gpus and max_epochs using regex
    modified_content = re.sub(r'gpus:.*', 'gpus: [0]', content)
    modified_content = re.sub(r'max_epochs:.*', 'max_epochs: 1', modified_content)
    # Handle max_nb_epochs as well (for vampvae.yaml)
    modified_content = re.sub(r'max_nb_epochs:.*', 'max_nb_epochs: 1', modified_content)

    # Write to a temporary file in the tmp subdirectory
    tmp_file_path = os.path.join(tmp_config_dir, f"tmp_{file_name}")
    with open(tmp_file_path, 'w') as f:
        f.write(modified_content)

    print(f"Created temporary config: {tmp_file_path}")

print("\nAll temporary configuration files have been created.")
