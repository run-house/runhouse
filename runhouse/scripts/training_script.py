"""Mock python job to run on remote server"""
import time

from runhouse.utils.utils import create_directory

DEST_DIR = 'training_folder'

print("Starting model training")

create_directory(DEST_DIR)

time.sleep(5)

print(f"Finished training - saved results to {DEST_DIR}")
