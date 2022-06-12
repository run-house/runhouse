"""Mock python job to run on remote server or be initialized as a URI"""
import time
import os


def bert_preprocessing():
    DEST_DIR = 'training_folder_bert'
    print("Starting model training")
    if not os.path.exists(DEST_DIR):
        os.makedirs(DEST_DIR)
    time.sleep(5)
    print(f"Finished training - saved results to {DEST_DIR}")
