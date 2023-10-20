Distributed: HF Accelerate
==========================

.. raw:: html

    <p><a href="https://colab.research.google.com/github/run-house/runhouse/blob/stable/docs/notebooks/examples/distributed.ipynb">
    <img height="20px" width="117px" src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a></p>


This tutorial demonstrates how to use Runhouse with HuggingFace
accelerate to launch distributed code on **your own remote hardware**.
We also show how one can reproducibly perform hardware dependency
autosetup, to ensure that your code runs smoothly every time.

You can run this on your own cluster, or through a standard cloud
account (AWS, GCP, Azure, LambdaLabs). If you do not have any compute or
cloud accounts set up, we recommend creating a
`LambdaLabs <https://cloud.lambdalabs.com/>`__ account for the easiest
setup path.

Install dependencies
--------------------

.. code:: ipython3

    !pip install accelerate
    !pip install runhouse

.. code:: ipython3

    import runhouse as rh


.. parsed-literal::
    :class: code-output

    INFO | 2023-03-20 17:56:13,023 | No auth token provided, so not using RNS API to save and load configs
    INFO | 2023-03-20 17:56:14,334 | NumExpr defaulting to 2 threads.


Setting up the Cluster
----------------------

On-Demand Cluster (AWS, Azure, GCP, or LambdaLabs)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For instructions on setting up cloud access for on-demand clusters,
please refer to `Cluster
Setup <https://www.run.house/docs/tutorials/quick_start#cluster-setup>`__.

.. code:: ipython3

    # single V100 GPU
    # gpu = rh.ondemand_cluster(name="rh-v100", instance_type="V100:1").up_if_not()

    # multigpu: 4 V100s
    gpu = rh.ondemand_cluster(name="rh-4-v100", instance_type="V100:4").up_if_not()

    # Set GPU to autostop after 60 min of inactivity (default is 30 min)
    gpu.keep_warm(60)  # or -1 to keep up indefinitely



On-Premise Cluster
~~~~~~~~~~~~~~~~~~

For an on-prem cluster, you can instantaite it as follows, filling in
the IP address, ssh user and private key path.

.. code:: ipython3

    # For an existing cluster
    # gpu = rh.cluster(ips=['<ip of the cluster>'],
    #                  ssh_creds={'ssh_user': '...', 'ssh_private_key':'<path_to_key>'},
    #                  name='rh-cluster')

Setting up Functions on Remote Hardware
---------------------------------------

Training Function
~~~~~~~~~~~~~~~~~

For simplicity, let’s use the
`training_function <https://github.com/huggingface/accelerate/blob/main/examples/nlp_example.py#L114>`__
from
`accelerate/examples/nlp_example.py <https://github.com/huggingface/accelerate/blob/v0.15.0/examples/nlp_example.py>`__
to demonstrate how to run this function remotely.

In this case, because the function is available on GitHub, we can pass
in a string pointing to the GitHub function.

For local functions, for instance if we had ``nlp_example.py`` in our
directory, we can also simply import the function.

.. code:: ipython3

    # if nlp_example.py is in local directory
    # from nlp_example import training_function

    # if function is available on GitHub, use it's string representation
    training_function = "https://github.com/huggingface/accelerate/blob/v0.15.0/examples/nlp_example.py:training_function"

Next, define the dependencies necessary to run the imported training
function using accelerate.

.. code:: ipython3

    reqs = ['pip:./accelerate', 'transformers', 'datasets', 'evaluate','tqdm', 'scipy', 'scikit-learn', 'tensorboard',
            'torch --upgrade --extra-index-url https://download.pytorch.org/whl/cu117']

Now, we can put together the above components (gpu cluster, training
function, and dependencies) to create our train function on remote
hardware.

.. code:: ipython3

    train_function_gpu = rh.function(
                              fn=training_function,
                              system=gpu,
                              reqs=reqs,
                          )


.. parsed-literal::
    :class: code-output

    INFO | 2023-03-20 21:01:46,942 | Setting up Function on cluster.
    INFO | 2023-03-20 21:01:46,951 | Installing packages on cluster rh-v100: ['GitPackage: https://github.com/huggingface/accelerate.git@v0.15.0', 'pip:./accelerate', 'transformers', 'datasets', 'evaluate', 'tqdm', 'scipy', 'scikit-learn', 'tensorboard', 'torch --upgrade --extra-index-url https://download.pytorch.org/whl/cu117']
    INFO | 2023-03-20 21:02:02,988 | Function setup complete.


``train_function_gpu`` is a callable that can be used just like the
original ``training_function`` function in the NLP example, except that
it runs the function on the specified cluster/system instead.

Launch Helper Function
----------------------

Here we define a helper function for launching accelerate training, and
then send the function to run on our GPU as well

.. code:: ipython3

    def launch_training(training_function, *args):
        from accelerate.utils import PrepareForLaunch, patch_environment
        import torch

        num_processes = torch.cuda.device_count()
        print(f'Device count: {num_processes}')
        with patch_environment(world_size=num_processes, master_addr="127.0.01", master_port="29500",
                               mixed_precision=args[1].mixed_precision):
            launcher = PrepareForLaunch(training_function, distributed_type="MULTI_GPU")
            torch.multiprocessing.start_processes(launcher, args=args, nprocs=num_processes, start_method="spawn")

.. code:: ipython3

    launch_training_gpu = rh.function(fn=launch_training).to(gpu)


.. parsed-literal::
    :class: code-output

    INFO | 2023-03-20 19:56:15,257 | Writing out function function to /content/launch_training_fn.py as functions serialized in notebooks are brittle. Please make sure the function does not rely on any local variables, including imports (which should be moved inside the function body).
    INFO | 2023-03-20 19:56:15,262 | Setting up Function on cluster.
    INFO | 2023-03-20 19:56:15,265 | Copying local package content to cluster <rh-v100>
    INFO | 2023-03-20 19:56:20,623 | Installing packages on cluster rh-v100: ['./']
    INFO | 2023-03-20 19:56:20,753 | Function setup complete.


Launch Distributed Training
---------------------------

Now, we’re ready to launch distributed training on our self-hosted
hardware!

.. code:: ipython3

    import argparse

    # define basic train args and hyperparams
    train_args = argparse.Namespace(cpu=False, mixed_precision='fp16')
    hps = {"lr": 2e-5, "num_epochs": 3, "seed": 42, "batch_size": 16}

.. code:: ipython3

    launch_training_gpu(train_function_gpu, hps, train_args, stream_logs=True)


.. parsed-literal::
    :class: code-output

    INFO | 2023-03-20 20:11:45,415 | Running launch_training via gRPC
    INFO | 2023-03-20 20:11:45,718 | Time to send message: 0.3 seconds
    INFO | 2023-03-20 20:11:45,720 | Submitted remote call to cluster. Result or logs can be retrieved
     with run_key "launch_training_20230320_201145", e.g.
    `rh.cluster(name="~/rh-v100").get("launch_training_20230320_201145", stream_logs=True)` in python
    `runhouse logs "rh-v100" launch_training_20230320_201145` from the command line.
     or cancelled with
    `rh.cluster(name="~/rh-v100").cancel("launch_training_20230320_201145")` in python or
    `runhouse cancel "rh-v100" launch_training_20230320_201145` from the command line.
    :task_name:launch_training
    :task_name:launch_training
    INFO | 2023-03-20 20:11:46,328 | Loading config from local file /home/ubuntu/runhouse/runhouse/builtins/config.json
    INFO | 2023-03-20 20:11:46,328 | No auth token provided, so not using RNS API to save and load configs
    Device count: 1
    INFO | 2023-03-20 20:11:49,486 | Loading config from local file /home/ubuntu/runhouse/runhouse/builtins/config.json
    INFO | 2023-03-20 20:11:49,486 | No auth token provided, so not using RNS API to save and load configs
    INFO | 2023-03-20 20:11:49,844 | Appending /home/ubuntu/accelerate/examples to sys.path
    INFO | 2023-03-20 20:11:49,844 | Importing module nlp_example

    Downloading builder script:   0%|          | 0.00/5.75k [00:00<?, ?B/s]
    Downloading builder script: 100%|██████████| 5.75k/5.75k [00:00<00:00, 8.19MB/s]

    Downloading (…)okenizer_config.json:   0%|          | 0.00/29.0 [00:00<?, ?B/s]
    Downloading (…)okenizer_config.json: 100%|██████████| 29.0/29.0 [00:00<00:00, 65.9kB/s]

    Downloading (…)lve/main/config.json:   0%|          | 0.00/570 [00:00<?, ?B/s]
    Downloading (…)lve/main/config.json: 100%|██████████| 570/570 [00:00<00:00, 1.22MB/s]

    Downloading (…)solve/main/vocab.txt:   0%|          | 0.00/213k [00:00<?, ?B/s]
    Downloading (…)solve/main/vocab.txt: 100%|██████████| 213k/213k [00:00<00:00, 1.05MB/s]
    Downloading (…)solve/main/vocab.txt: 100%|██████████| 213k/213k [00:00<00:00, 1.05MB/s]

    Downloading (…)/main/tokenizer.json:   0%|          | 0.00/436k [00:00<?, ?B/s]
    Downloading (…)/main/tokenizer.json: 100%|██████████| 436k/436k [00:00<00:00, 1.61MB/s]
    Downloading (…)/main/tokenizer.json: 100%|██████████| 436k/436k [00:00<00:00, 1.60MB/s]

    Downloading builder script:   0%|          | 0.00/28.8k [00:00<?, ?B/s]
    Downloading builder script: 100%|██████████| 28.8k/28.8k [00:00<00:00, 380kB/s]

    Downloading metadata:   0%|          | 0.00/28.7k [00:00<?, ?B/s]
    Downloading metadata: 100%|██████████| 28.7k/28.7k [00:00<00:00, 422kB/s]

    Downloading readme:   0%|          | 0.00/27.9k [00:00<?, ?B/s]
    Downloading readme: 100%|██████████| 27.9k/27.9k [00:00<00:00, 412kB/s]

    Downloading data files:   0%|          | 0/3 [00:00<?, ?it/s]Downloading and preparing dataset glue/mrpc to /home/ubuntu/.cache/huggingface/datasets/glue/mrpc/1.0.0/dacbe3125aa31d7f70367a07a8a9e72a5a0bfeb5fc42e75c9db75b96da6053ad...


    Downloading data: 0.00B [00:00, ?B/s][A
    Downloading data: 6.22kB [00:00, 11.1MB/s]

    Downloading data files:  33%|███▎      | 1/3 [00:00<00:00,  4.26it/s]

    Downloading data: 0.00B [00:00, ?B/s][A
    Downloading data: 1.05MB [00:00, 55.0MB/s]

    Downloading data files:  67%|██████▋   | 2/3 [00:00<00:00,  5.30it/s]

    Downloading data: 0.00B [00:00, ?B/s][A
    Downloading data: 441kB [00:00, 44.3MB/s]

    Downloading data files: 100%|██████████| 3/3 [00:00<00:00,  5.87it/s]
    Downloading data files: 100%|██████████| 3/3 [00:00<00:00,  5.56it/s]

    Generating train split:   0%|          | 0/3668 [00:00<?, ? examples/s]
    Generating train split:  79%|███████▉  | 2898/3668 [00:00<00:00, 28934.98 examples/s]


    Generating validation split:   0%|          | 0/408 [00:00<?, ? examples/s]


    Generating test split:   0%|          | 0/1725 [00:00<?, ? examples/s]


      0%|          | 0/3 [00:00<?, ?it/s]
    100%|██████████| 3/3 [00:00<00:00, 1296.81it/s]

    Map:   0%|          | 0/3668 [00:00<?, ? examples/s]
    Map: 100%|██████████| 3668/3668 [00:00<00:00, 33587.18 examples/s]


    Map:   0%|          | 0/408 [00:00<?, ? examples/s]


    Map:   0%|          | 0/1725 [00:00<?, ? examples/s]

    Dataset glue downloaded and prepared to /home/ubuntu/.cache/huggingface/datasets/glue/mrpc/1.0.0/dacbe3125aa31d7f70367a07a8a9e72a5a0bfeb5fc42e75c9db75b96da6053ad. Subsequent calls will reuse this data.

    Downloading pytorch_model.bin:   0%|          | 0.00/436M [00:00<?, ?B/s]
    Downloading pytorch_model.bin:   2%|▏         | 10.5M/436M [00:00<00:04, 95.8MB/s]
    Downloading pytorch_model.bin:   5%|▍         | 21.0M/436M [00:00<00:04, 97.1MB/s]
    Downloading pytorch_model.bin:   7%|▋         | 31.5M/436M [00:00<00:04, 93.2MB/s]
    Downloading pytorch_model.bin:  10%|▉         | 41.9M/436M [00:00<00:04, 91.3MB/s]
    Downloading pytorch_model.bin:  12%|█▏        | 52.4M/436M [00:00<00:04, 92.6MB/s]
    Downloading pytorch_model.bin:  14%|█▍        | 62.9M/436M [00:00<00:04, 86.0MB/s]
    Downloading pytorch_model.bin:  17%|█▋        | 73.4M/436M [00:00<00:04, 89.9MB/s]
    Downloading pytorch_model.bin:  19%|█▉        | 83.9M/436M [00:00<00:03, 90.2MB/s]
    Downloading pytorch_model.bin:  22%|██▏       | 94.4M/436M [00:01<00:03, 91.5MB/s]
    Downloading pytorch_model.bin:  24%|██▍       | 105M/436M [00:01<00:03, 93.3MB/s]
    Downloading pytorch_model.bin:  26%|██▋       | 115M/436M [00:01<00:03, 86.5MB/s]
    Downloading pytorch_model.bin:  29%|██▉       | 126M/436M [00:01<00:03, 86.9MB/s]
    Downloading pytorch_model.bin:  31%|███▏      | 136M/436M [00:01<00:03, 87.2MB/s]
    Downloading pytorch_model.bin:  34%|███▎      | 147M/436M [00:01<00:03, 88.6MB/s]
    Downloading pytorch_model.bin:  36%|███▌      | 157M/436M [00:01<00:03, 90.7MB/s]
    Downloading pytorch_model.bin:  38%|███▊      | 168M/436M [00:01<00:02, 90.4MB/s]
    Downloading pytorch_model.bin:  41%|████      | 178M/436M [00:02<00:03, 82.5MB/s]
    Downloading pytorch_model.bin:  43%|████▎     | 189M/436M [00:02<00:02, 84.6MB/s]
    Downloading pytorch_model.bin:  46%|████▌     | 199M/436M [00:02<00:02, 81.3MB/s]
    Downloading pytorch_model.bin:  48%|████▊     | 210M/436M [00:02<00:02, 84.4MB/s]
    Downloading pytorch_model.bin:  51%|█████     | 220M/436M [00:02<00:02, 83.4MB/s]
    Downloading pytorch_model.bin:  53%|█████▎    | 231M/436M [00:02<00:02, 86.4MB/s]
    Downloading pytorch_model.bin:  55%|█████▌    | 241M/436M [00:02<00:02, 88.9MB/s]
    Downloading pytorch_model.bin:  58%|█████▊    | 252M/436M [00:02<00:02, 90.9MB/s]
    Downloading pytorch_model.bin:  60%|██████    | 262M/436M [00:02<00:01, 91.6MB/s]
    Downloading pytorch_model.bin:  63%|██████▎   | 273M/436M [00:03<00:01, 90.9MB/s]
    Downloading pytorch_model.bin:  65%|██████▍   | 283M/436M [00:03<00:01, 90.8MB/s]
    Downloading pytorch_model.bin:  67%|██████▋   | 294M/436M [00:03<00:01, 91.6MB/s]
    Downloading pytorch_model.bin:  70%|██████▉   | 304M/436M [00:03<00:01, 92.1MB/s]
    Downloading pytorch_model.bin:  72%|███████▏  | 315M/436M [00:03<00:01, 91.9MB/s]
    Downloading pytorch_model.bin:  75%|███████▍  | 325M/436M [00:03<00:01, 91.0MB/s]
    Downloading pytorch_model.bin:  77%|███████▋  | 336M/436M [00:03<00:01, 89.7MB/s]
    Downloading pytorch_model.bin:  79%|███████▉  | 346M/436M [00:03<00:00, 90.2MB/s]
    Downloading pytorch_model.bin:  82%|████████▏ | 357M/436M [00:03<00:00, 92.1MB/s]
    Downloading pytorch_model.bin:  84%|████████▍ | 367M/436M [00:04<00:00, 93.5MB/s]
    Downloading pytorch_model.bin:  87%|████████▋ | 377M/436M [00:04<00:00, 93.5MB/s]
    Downloading pytorch_model.bin:  89%|████████▉ | 388M/436M [00:04<00:00, 92.9MB/s]
    Downloading pytorch_model.bin:  91%|█████████▏| 398M/436M [00:04<00:00, 81.5MB/s]
    Downloading pytorch_model.bin:  94%|█████████▍| 409M/436M [00:04<00:00, 83.7MB/s]
    Downloading pytorch_model.bin:  96%|█████████▌| 419M/436M [00:04<00:00, 85.6MB/s]
    Downloading pytorch_model.bin:  99%|█████████▊| 430M/436M [00:04<00:00, 80.6MB/s]
    Downloading pytorch_model.bin: 100%|██████████| 436M/436M [00:04<00:00, 88.2MB/s]
    Some weights of the model checkpoint at bert-base-cased were not used when initializing BertForSequenceClassification: ['cls.predictions.transform.LayerNorm.bias', 'cls.predictions.transform.dense.weight', 'cls.seq_relationship.bias', 'cls.predictions.transform.LayerNorm.weight', 'cls.predictions.transform.dense.bias', 'cls.predictions.decoder.weight', 'cls.seq_relationship.weight', 'cls.predictions.bias']
    - This IS expected if you are initializing BertForSequenceClassification from the checkpoint of a model trained on another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPreTraining model).
    - This IS NOT expected if you are initializing BertForSequenceClassification from the checkpoint of a model that you expect to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model).
    Some weights of BertForSequenceClassification were not initialized from the model checkpoint at bert-base-cased and are newly initialized: ['classifier.weight', 'classifier.bias']
    You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
    You're using a BertTokenizerFast tokenizer. Please note that with a fast tokenizer, using the `__call__` method is faster than using a method to encode the text followed by a call to the `pad` method to get a padded encoding.
    epoch 0: {'accuracy': 0.7745098039215687, 'f1': 0.8557993730407523}
    epoch 1: {'accuracy': 0.8406862745098039, 'f1': 0.8849557522123894}
    epoch 2: {'accuracy': 0.8553921568627451, 'f1': 0.8981001727115717}


Terminate Cluster
-----------------

Once you are done using the cluster, you can terminate it as follows:

.. code:: ipython3

    gpu.teardown()
