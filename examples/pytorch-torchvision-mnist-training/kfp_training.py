# # A Kubeflow Training Pipeline with Kubetorch

# This example demonstrates how to use Kubeflow to orchestrate the training of a basic Torch model,
# with the training class dispatched to a GPU-enabled pod to actually do the training.
#
# We use the very popular MNIST dataset which includes a large number
# of handwritten digits, and create a neural network that accurately identifies
# what digit is in an image.
#
# ## Setting up the Kubeflow pipeline
# The Kubeflow pipeline is extremely lean, with each step simply wrapping the steps in torch_basic_example.py
# to automate and observe them as a pipeline.

import kfp
from kfp.dsl import component, pipeline

S3_BUCKET = "s3://my-simple-torch-model-example/"

# First we download and lightly preprocess the dataset. The MNIST example is trivial, but we are doing this
# preprocessing on separate compute from training purely for demonstration's sake.
@component(base_image="pypypypy/my-pipeline-image:latest")
def preprocess():
    import kubetorch as kt

    from torch_basic_example import preprocess_data

    cpus = kt.Compute(cpus=16, image=kt.images.pytorch())
    preproc = kt.fn(preprocess_data).to(cpus)
    s3_path = f"{S3_BUCKET}/{kfp.dsl.PIPELINE_JOB_NAME_PLACEHOLDER}/data"
    preproc(s3_path)


# Now we run the training. In this step, we dispatch the training to the remote cluster. The model is trained on the remote cluster, and the model checkpoints are saved to an S3 bucket.
@component(base_image="pypypypy/my-pipeline-image:latest")
def train_model():
    import kubetorch as kt

    from torch_basic_example import SimpleTrainer

    gpu = kt.Compute(gpus="A10G:1", image=kt.images.pytorch())

    model = kt.cls(SimpleTrainer).to(gpu)

    # We set some settings for the model training
    batch_size = 64
    epochs = 5
    learning_rate = 0.01

    s3_path = f"{S3_BUCKET}/{kfp.dsl.PIPELINE_JOB_NAME_PLACEHOLDER}"
    model.load_train(f"{s3_path}/data", batch_size, download=True)
    model.load_test(f"{s3_path}/data", batch_size, download=True)

    # We can train the model per epoch, use the remote .test() method to assess the accuracy, and save it from remote to a S3 bucket.
    # All errors, prints, and logs are sent to me as if I were debugging on my local machine, but all the work is done in the cloud.
    for epoch in range(epochs):
        # Train one epoch and test the model
        model.train_model(learning_rate=learning_rate)
        model.test_model()

        # Save each model checkpoint to S3
        model.save_model(
            bucket_name=S3_BUCKET,
            s3_file_path=f"{kfp.dsl.PIPELINE_JOB_NAME_PLACEHOLDER}/checkpoints/model_epoch_{epoch + 1}.pth",
        )

    # Save the final model
    model.save_model(
        bucket_name=S3_BUCKET,
        s3_file_path=f"{kfp.dsl.PIPELINE_JOB_NAME_PLACEHOLDER}/checkpoints/model_final.pth",
    )


# Deploy the model for inference
@component(base_image="pypypypy/my-pipeline-image:latest")
def deploy_inference():
    import kubetorch as kt

    from torch_basic_example import SimpleTrainer

    checkpoint_path = f"{S3_BUCKET}/{kfp.dsl.PIPELINE_JOB_NAME_PLACEHOLDER}/checkpoints/model_final.pth"
    local_checkpoint_path = "/model.pth"
    # Download the model checkpoint into the image for fast cold starts
    img = kt.images.pytorch().run_bash(
        f"aws s3 cp {checkpoint_path} {local_checkpoint_path}"
    )
    inference_compute = kt.Compute(
        gpus="A10G:1",
        image=img,
    )

    init_args = dict(from_checkpoint=local_checkpoint_path)
    model = kt.cls(SimpleTrainer).to(inference_compute, init_args=init_args)

    print(f"Model endpoint: {model.predict.endpoint()}")


# Define the pipeline. This is a simple linear pipeline with two steps: preprocess the data, train the model, and deploy for inference.
@pipeline(
    name="PyTorch Training Pipeline",
    description="A simple PyTorch training pipeline with multiple steps",
)
def pytorch_training_pipeline():
    preprocess_task = preprocess()
    train_task = train_model()
    deploy_task = deploy_inference()

    train_task.after(preprocess_task)
    deploy_task.after(train_task)


# Compile the pipeline
kfp.compiler.Compiler().compile(
    pytorch_training_pipeline, "pytorch_training_pipeline.yaml"
)
