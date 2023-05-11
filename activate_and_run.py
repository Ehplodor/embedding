# Note: The script assumes that you have the virtualenv package installed. If you don't have it, you can install it by running pip install virtualenv.
# When you run this script, it will check if the virtual environment exists in the specified directory. If it doesn't exist, it will initialize the virtual environment using virtualenv. It will then activate the virtual environment and execute the main script using the Python interpreter from the virtual environment.
# This script is compatible with Windows, but you may need to modify the venv_activate path for non-Windows systems.

import os
import subprocess

# Define the paths
venv_dir = 'venv'  # Specify the name of the virtual environment directory
main_script = 'word_embedding_trainer.py'  # Specify the name of your main script

# Check if venv exists
if not os.path.exists(venv_dir):
    # Initialize venv using virtualenv
    subprocess.run(['python', '-m', 'virtualenv', venv_dir], check=True)

# Activate venv
venv_activate = os.path.join(venv_dir, 'Scripts', 'activate.bat')  # Adjust for non-Windows systems
activate_cmd = f'call "{venv_activate}"'
subprocess.run(activate_cmd, shell=True, check=True)

# Execute the main script
subprocess.run(['python', main_script], check=True)
