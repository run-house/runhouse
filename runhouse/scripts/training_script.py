"""Mock python job to run on remote server"""
import time

from runhouse.utils.utils import create_directory


def bert_preprocessing():
    DEST_DIR = 'training_folder_bert'
    print("Starting model training")
    create_directory(DEST_DIR)
    time.sleep(5)
    print(f"Finished training - saved results to {DEST_DIR}")


if __name__ == "__main__":
    print("Running bert preprocessing")
    bert_preprocessing()
