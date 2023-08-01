# Launch an instance by creating an indefinite training job
import sagemaker_ssh_helper

sagemaker_ssh_helper.setup_and_start_ssh()

if __name__ == "__main__":
    import time

    while True:
        time.sleep(10)
