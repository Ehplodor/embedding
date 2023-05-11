import subprocess

# Step 1: Install pipreqs
subprocess.run(['pip', 'install', 'pipreqs'], check=True)

# Step 2: Generate requirements.txt using pipreqs
subprocess.run(['pipreqs', '--force'], check=True)

print("Requirements update completed.")
