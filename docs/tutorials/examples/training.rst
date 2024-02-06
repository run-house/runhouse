Training: Transformers
======================

.. raw:: html

    <p><a href="https://colab.research.google.com/github/run-house/runhouse/blob/stable/docs/notebooks/examples/training.ipynb">
    <img height="20px" width="117px" src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a></p>

This tutorial demonstrates how to use Runhouse to facilitate model
training on **your own GPU**. With Runhouse, easily run your local code
or training script on a remote cluster, and reproducibly set up your
remote training environment.

You can run this on your own cluster, or through a standard cloud
account (AWS, GCP, Azure, LambdaLabs). If you do not have any compute or
cloud accounts set up, we recommend creating a
`LambdaLabs <https://cloud.lambdalabs.com/>`__ account for the easiest
setup path.

Table of Contents
-----------------

-  Hardware Setup
-  Dataloading and Preprocessing
-  Model Training

Install Runhouse
----------------

.. code:: ipython3

    !pip install runhouse[sky]

.. code:: ipython3

    import runhouse as rh


.. parsed-literal::
    :class: code-output

    INFO | 2023-06-08 18:12:34,980 | No auth token provided, so not using RNS API to save and load configs
    INFO | 2023-06-08 18:12:36,499 | NumExpr defaulting to 2 threads.


Hardware Setup
--------------

If you’re not already familiar with setting up a Runhouse cluster,
please first refer to `Cluster
Setup <https://www.run.house/docs/tutorials/quick_start#cluster-setup>`__
for a more introductory and in-depth walkthrough.

.. code:: ipython3

    # Optional, to sync over any hardware credentials and configs from your Runhouse account
    !runhouse login --yes

    # alternatively, to set up credentials locally, run `!sky check` and follow the instructions for your cloud provider(s)
    # !sky check


.. code:: ipython3

    # sample on-demand cluster, launched through Runhouse/SkyPilot
    gpu = rh.ondemand_cluster(name='rh-a10x', instance_type='g5.2xlarge', provider='aws').up_if_not()

    # or for your own dedicated cluster
    # gpu = rh.cluster(
    #            name="cpu-cluster",
    #            ips=['<ip of the cluster>'],
    #            ssh_creds={'ssh_user': '<user>', 'ssh_private_key':'<path_to_key>'},
    #       )


.. parsed-literal::
    :class: code-output

    INFO | 2023-06-08 18:30:56,926 | Attempting to load config for /carolineechen/rh-a10x from RNS.



.. parsed-literal::
    :class: code-output

    Output()



Dataloading and Preprocessing
-----------------------------

Here, we briefly demonstrate data loading and preprocessing on our remote hardware.

Steps:

- take our preprocessing code, wrap it in a function called load_and_preprocess

- create a runhouse function, send it along w/ dependencies to the cluster, auto set up is handled

- call the function (which runs remotely on the cluster!)

Note that all the code inside the function runs on our gpu cluster,
which means there’s no need to install anything locally either.

For a more in-depth walkthrough of Runhouse’s function and env APIs,
please refer to the `Compute API
Tutorial <https://www.run.house/docs/tutorials/api/compute>`__.

.. code:: ipython3

    def load_and_preprocess():
        from datasets import load_dataset

        dataset = load_dataset("yelp_review_full")
        dataset["train"][100]

        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

        def tokenize_function(examples):
            return tokenizer(examples["text"], padding="max_length", truncation=True)

        tokenized_datasets = dataset.map(tokenize_function, batched=True)

        small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
        small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(1000))
        return [small_train_dataset, small_eval_dataset]

.. code:: ipython3

    reqs = ["transformers", "datasets", "torch"]

    load_and_preprocess = rh.function(fn=load_and_preprocess).to(gpu, env=reqs)


.. parsed-literal::
    :class: code-output

    INFO | 2023-06-08 18:43:59,993 | Writing out function function to /content/load_and_preprocess_fn.py. Please make sure the function does not rely on any local variables, including imports (which should be moved inside the function body).
    INFO | 2023-06-08 18:44:00,000 | Setting up Function on cluster.
    INFO | 2023-06-08 18:44:00,478 | Connected (version 2.0, client OpenSSH_8.2p1)
    INFO | 2023-06-08 18:44:00,684 | Authentication (publickey) successful!
    INFO | 2023-06-08 18:44:07,003 | Installing packages on cluster rh-a10x: ['transformers', 'datasets', 'torch', 'Package: content']
    INFO | 2023-06-08 18:46:10,042 | Function setup complete.


Runhouse functions work so that you call them as you would with a local
function (e.g. ``data = load_and_preprocess()``) – the code runs
remotely and returns the object locally.

However, in this case, as we are running training on the same cluster
and it’s not useful to have the dataset sent back to local, we can
simply call ``.remote()`` on the function to have it run async,
returning an object reference to our dataset rather than the actual
data. This dataset ref can be passed into later functions as if they
were the actual object.

If you’d like to save down your data to file storage (e.g. ``s3``,
``gcs``), Runhouse also has API support for that. Please refer to our
Data API Tutorial for more information on that.

.. code:: ipython3

    datasets_ref = load_and_preprocess.remote()


.. parsed-literal::
    :class: code-output

    INFO | 2023-06-08 18:52:55,092 | Running load_and_preprocess via HTTP
    INFO | 2023-06-08 18:52:55,191 | Time to call remote function: 0.1 seconds
    INFO | 2023-06-08 18:52:55,193 | Submitted remote call to cluster. Result or logs can be retrieved
     with run_key "load_and_preprocess_20230608_185255", e.g.
    `rh.cluster(name="/carolineechen/rh-a10x").get("load_and_preprocess_20230608_185255", stream_logs=True)` in python
    `runhouse logs "rh-a10x" load_and_preprocess_20230608_185255` from the command line.
     or cancelled with
    `rh.cluster(name="/carolineechen/rh-a10x").cancel("load_and_preprocess_20230608_185255")` in python or
    `runhouse cancel "rh-a10x" load_and_preprocess_20230608_185255` from the command line.


Training
--------

Now that we have the dataset ready, it’s time to train!

In a similar flow as above: - take our training code, wrap it in a
``train`` function - specify the function and relevant dependencies to
be synced and installed on the remote cluster - call the function from
local, passing in your dataset reference, and watch it train remotely

Later on, we also demonstrate how you can run training from an existing
script.

Training from locally defined functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    def train(hf_datasets):
        [small_train_dataset, small_eval_dataset] = hf_datasets

        from transformers import AutoModelForSequenceClassification

        model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=5)

        import numpy as np
        import evaluate

        metric = evaluate.load("accuracy")  # Requires scikit-learn

        def compute_metrics(eval_pred):
            logits, labels = eval_pred
            predictions = np.argmax(logits, axis=-1)
            return metric.compute(predictions=predictions, references=labels)

        from transformers import TrainingArguments, Trainer

        training_args = TrainingArguments(output_dir="test_trainer", evaluation_strategy="epoch")

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=small_train_dataset,
            eval_dataset=small_eval_dataset,
            compute_metrics=compute_metrics,
        )

        trainer.train()

.. code:: ipython3

    extra_reqs = ['evaluate', 'scikit-learn', 'accelerate']

    train = rh.function(fn=train).to(gpu, env=extra_reqs)


.. parsed-literal::
    :class: code-output

    INFO | 2023-06-08 18:53:03,726 | Writing out function function to /content/train_fn.py. Please make sure the function does not rely on any local variables, including imports (which should be moved inside the function body).
    INFO | 2023-06-08 18:53:03,730 | Setting up Function on cluster.
    INFO | 2023-06-08 18:53:05,568 | Installing packages on cluster rh-a10x: ['evaluate', 'scikit-learn', 'accelerate', 'Package: content']
    INFO | 2023-06-08 18:53:17,394 | Function setup complete.


To run the function, call it as you would any Python function. Pass in the dataset reference, and optionally add `stream_logs=True` to see the logs locally.

.. code:: ipython3

    train(datasets_ref, stream_logs=True)


.. parsed-literal::
    :class: code-output

    INFO | 2023-06-08 18:53:21,114 | Running train via HTTP
    INFO | 2023-06-08 18:56:10,362 | Time to call remote function: 169.25 seconds
    INFO | 2023-06-08 18:56:10,365 | Submitted remote call to cluster. Result or logs can be retrieved
     with run_key "train_20230608_185610", e.g.
    `rh.cluster(name="/carolineechen/rh-a10x").get("train_20230608_185610", stream_logs=True)` in python
    `runhouse logs "rh-a10x" train_20230608_185610` from the command line.
     or cancelled with
    `rh.cluster(name="/carolineechen/rh-a10x").cancel("train_20230608_185610")` in python or
    `runhouse cancel "rh-a10x" train_20230608_185610` from the command line.
    :job_id:01000000
    :task_name:get_fn_from_pointers
    :job_id:01000000
    INFO | 2023-06-08 18:56:11,007 | Loaded Runhouse config from /home/ubuntu/.rh/config.yaml
    :task_name:get_fn_from_pointers
    INFO | 2023-06-08 18:56:11,821 | Writing logs on cluster to /home/ubuntu/.rh/logs/train_20230608_185610
    INFO | 2023-06-08 18:56:11,821 | Appending /home/ubuntu/content to sys.path
    INFO | 2023-06-08 18:56:11,821 | Importing module train_fn

    Downloading model.safetensors:   0%|          | 0.00/436M [00:00<?, ?B/s]
    Downloading model.safetensors:  12%|█▏        | 52.4M/436M [00:00<00:00, 468MB/s]
    Downloading model.safetensors:  24%|██▍       | 105M/436M [00:00<00:00, 490MB/s]
    Downloading model.safetensors:  36%|███▌      | 157M/436M [00:00<00:00, 447MB/s]
    Downloading model.safetensors:  48%|████▊     | 210M/436M [00:00<00:00, 446MB/s]
    Downloading model.safetensors:  60%|██████    | 262M/436M [00:00<00:00, 448MB/s]
    Downloading model.safetensors:  72%|███████▏  | 315M/436M [00:00<00:00, 456MB/s]
    Downloading model.safetensors:  84%|████████▍ | 367M/436M [00:00<00:00, 472MB/s]
    Downloading model.safetensors:  96%|█████████▋| 419M/436M [00:00<00:00, 480MB/s]
    Downloading model.safetensors: 100%|██████████| 436M/436M [00:00<00:00, 467MB/s]
    Some weights of the model checkpoint at bert-base-cased were not used when initializing BertForSequenceClassification: ['cls.predictions.bias', 'cls.seq_relationship.weight', 'cls.predictions.transform.dense.bias', 'cls.seq_relationship.bias', 'cls.predictions.transform.LayerNorm.weight', 'cls.predictions.transform.LayerNorm.bias', 'cls.predictions.transform.dense.weight']
    - This IS expected if you are initializing BertForSequenceClassification from the checkpoint of a model trained on another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPreTraining model).
    - This IS NOT expected if you are initializing BertForSequenceClassification from the checkpoint of a model that you expect to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model).
    Some weights of BertForSequenceClassification were not initialized from the model checkpoint at bert-base-cased and are newly initialized: ['classifier.bias', 'classifier.weight']
    You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.

    Downloading builder script:   0%|          | 0.00/4.20k [00:00<?, ?B/s]
    Downloading builder script: 100%|██████████| 4.20k/4.20k [00:00<00:00, 7.13MB/s]

      0%|          | 0/375 [00:00<?, ?it/s]
      0%|          | 1/375 [01:11<7:28:39, 71.98s/it]
      1%|          | 2/375 [01:12<3:05:04, 29.77s/it]
      1%|          | 3/375 [01:12<1:40:56, 16.28s/it]
      1%|          | 4/375 [01:12<1:01:29,  9.94s/it]
      1%|▏         | 5/375 [01:12<39:42,  6.44s/it]
      2%|▏         | 6/375 [01:13<26:37,  4.33s/it]
      2%|▏         | 7/375 [01:13<18:19,  2.99s/it]
      2%|▏         | 8/375 [01:13<12:54,  2.11s/it]
      2%|▏         | 9/375 [01:13<09:16,  1.52s/it]
      3%|▎         | 10/375 [01:14<06:49,  1.12s/it]
    [... truncated output ...]
     31%|███       | 115/375 [01:37<00:59,  4.38it/s]
     31%|███       | 116/375 [01:38<00:59,  4.39it/s]
     31%|███       | 117/375 [01:38<00:58,  4.38it/s]
     31%|███▏      | 118/375 [01:38<00:58,  4.38it/s]
     32%|███▏      | 119/375 [01:38<00:58,  4.38it/s]
     32%|███▏      | 120/375 [01:39<00:58,  4.38it/s]
     32%|███▏      | 121/375 [01:39<00:57,  4.38it/s]
     33%|███▎      | 122/375 [01:39<00:57,  4.38it/s]
     33%|███▎      | 123/375 [01:39<00:57,  4.38it/s]
     33%|███▎      | 124/375 [01:40<00:57,  4.38it/s]
     33%|███▎      | 125/375 [01:40<00:57,  4.38it/s]

      0%|          | 0/125 [00:00<?, ?it/s][A

      2%|▏         | 3/125 [00:00<00:06, 19.31it/s][A

      4%|▍         | 5/125 [00:00<00:07, 15.52it/s][A

      6%|▌         | 7/125 [00:00<00:08, 14.32it/s][A

    [... truncated output ...]

     90%|█████████ | 113/125 [00:08<00:00, 12.87it/s][A

     92%|█████████▏| 115/125 [00:08<00:00, 12.87it/s][A

     94%|█████████▎| 117/125 [00:09<00:00, 12.87it/s][A

     95%|█████████▌| 119/125 [00:09<00:00, 12.86it/s][A

     97%|█████████▋| 121/125 [00:09<00:00, 12.86it/s][A

     98%|█████████▊| 123/125 [00:09<00:00, 12.86it/s][A

    100%|██████████| 125/125 [00:09<00:00, 12.87it/s][A

    [A
     33%|███▎      | 125/375 [01:50<00:57,  4.38it/s]

    100%|██████████| 125/125 [00:09<00:00, 12.87it/s][A

                                                     [A

     34%|███▎      | 126/375 [01:50<13:03,  3.15s/it]{'eval_loss': 1.1413816213607788, 'eval_accuracy': 0.518, 'eval_runtime': 9.7297, 'eval_samples_per_second': 102.778, 'eval_steps_per_second': 12.847, 'epoch': 1.0}

     34%|███▍      | 127/375 [01:50<09:23,  2.27s/it]
     34%|███▍      | 128/375 [01:50<06:49,  1.66s/it]
     34%|███▍      | 129/375 [01:50<05:02,  1.23s/it]
     35%|███▍      | 130/375 [01:51<03:47,  1.08it/s]
     35%|███▍      | 131/375 [01:51<02:55,  1.39it/s]
     35%|███▌      | 132/375 [01:51<02:18,  1.75it/s]
     35%|███▌      | 133/375 [01:51<01:53,  2.13it/s]
     36%|███▌      | 134/375 [01:52<01:35,  2.52it/s]
    [... truncated output ...]
     64%|██████▍   | 240/375 [02:16<00:30,  4.38it/s]
     64%|██████▍   | 241/375 [02:16<00:30,  4.38it/s]
     65%|██████▍   | 242/375 [02:16<00:30,  4.37it/s]
     65%|██████▍   | 243/375 [02:16<00:30,  4.38it/s]
     65%|██████▌   | 244/375 [02:17<00:29,  4.38it/s]
     65%|██████▌   | 245/375 [02:17<00:29,  4.37it/s]
     66%|██████▌   | 246/375 [02:17<00:29,  4.38it/s]
     66%|██████▌   | 247/375 [02:17<00:29,  4.38it/s]
     66%|██████▌   | 248/375 [02:18<00:29,  4.37it/s]
     66%|██████▋   | 249/375 [02:18<00:28,  4.38it/s]
     67%|██████▋   | 250/375 [02:18<00:28,  4.38it/s]

      0%|          | 0/125 [00:00<?, ?it/s][A

      2%|▏         | 3/125 [00:00<00:06, 19.24it/s][A

      4%|▍         | 5/125 [00:00<00:07, 15.45it/s][A

      6%|▌         | 7/125 [00:00<00:08, 14.28it/s][A

      7%|▋         | 9/125 [00:00<00:08, 13.72it/s][A

      9%|▉         | 11/125 [00:00<00:08, 13.39it/s][A

    [... truncated output ...]

     92%|█████████▏| 115/125 [00:08<00:00, 12.83it/s][A

     94%|█████████▎| 117/125 [00:09<00:00, 12.84it/s][A

     95%|█████████▌| 119/125 [00:09<00:00, 12.83it/s][A

     97%|█████████▋| 121/125 [00:09<00:00, 12.82it/s][A

     98%|█████████▊| 123/125 [00:09<00:00, 12.81it/s][A

    100%|██████████| 125/125 [00:09<00:00, 12.83it/s][A



    [A
     67%|██████▋   | 250/375 [02:28<00:28,  4.38it/s]

    100%|██████████| 125/125 [00:09<00:00, 12.83it/s][A

                                                     [A{'eval_loss': 1.0151797533035278, 'eval_accuracy': 0.576, 'eval_runtime': 9.7523, 'eval_samples_per_second': 102.539, 'eval_steps_per_second': 12.817, 'epoch': 2.0}

     67%|██████▋   | 251/375 [02:28<06:34,  3.18s/it]
     67%|██████▋   | 252/375 [02:28<04:42,  2.29s/it]
     67%|██████▋   | 253/375 [02:29<03:24,  1.67s/it]
     68%|██████▊   | 254/375 [02:29<02:30,  1.24s/it]
     68%|██████▊   | 255/375 [02:29<01:52,  1.07it/s]
    [... truncated output ...]
     99%|█████████▉| 371/375 [02:56<00:00,  4.37it/s]
     99%|█████████▉| 372/375 [02:56<00:00,  4.37it/s]
     99%|█████████▉| 373/375 [02:56<00:00,  4.37it/s]
    100%|█████████▉| 374/375 [02:56<00:00,  4.37it/s]
    100%|██████████| 375/375 [02:56<00:00,  4.38it/s]

      0%|          | 0/125 [00:00<?, ?it/s][A

      2%|▏         | 3/125 [00:00<00:06, 19.19it/s][A

      4%|▍         | 5/125 [00:00<00:07, 15.41it/s][A

      6%|▌         | 7/125 [00:00<00:08, 14.22it/s][A

      7%|▋         | 9/125 [00:00<00:08, 13.69it/s][A

      9%|▉         | 11/125 [00:00<00:08, 13.37it/s][A

     10%|█         | 13/125 [00:00<00:08, 13.18it/s][A

     12%|█▏        | 15/125 [00:01<00:08, 13.06it/s][A

    [... truncated output ...]

     89%|████████▉ | 111/125 [00:08<00:01, 12.82it/s][A

     90%|█████████ | 113/125 [00:08<00:00, 12.83it/s][A

     92%|█████████▏| 115/125 [00:08<00:00, 12.82it/s][A

     94%|█████████▎| 117/125 [00:09<00:00, 12.82it/s][A

     95%|█████████▌| 119/125 [00:09<00:00, 12.83it/s][A

     97%|█████████▋| 121/125 [00:09<00:00, 12.83it/s][A

     98%|█████████▊| 123/125 [00:09<00:00, 12.83it/s][A

    100%|██████████| 125/125 [00:09<00:00, 12.83it/s][A



    [A
    100%|██████████| 375/375 [03:06<00:00,  4.38it/s]

    100%|██████████| 125/125 [00:09<00:00, 12.83it/s][A

                                                     [A


    100%|██████████| 375/375 [03:06<00:00,  4.38it/s]
    100%|██████████| 375/375 [03:06<00:00,  2.01it/s]
    {'eval_loss': 1.0775768756866455, 'eval_accuracy': 0.568, 'eval_runtime': 9.7574, 'eval_samples_per_second': 102.486, 'eval_steps_per_second': 12.811, 'epoch': 3.0}
    {'train_runtime': 186.7323, 'train_samples_per_second': 16.066, 'train_steps_per_second': 2.008, 'train_loss': 0.98061328125, 'epoch': 3.0}


Training from existing script
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Runhouse also makes it easy to run scripts and commands on your remote
cluster, so if you have an existing training script, you can easily
directly run that on your remote compute as well.

-  Sync over your working directory with the training script to the
   cluster
-  Set up environment and package installations on the cluster
-  Run the script with a simple command

To sync over the working directory, you can create a Runhouse folder
resource and send it over to the cluster.

.. code:: ipython3

    rh.folder(path="local_folder_path", dest_path="remote_folder_path").to(gpu)

Alternatively, if the script lives inside a GitHub repo, you could also
directly clone and install the GitHub repo remotely with the GitPackage
API.

In this case, let’s say we’re trying to access and run
`examples/nlp_example.py <https://github.com/huggingface/accelerate/blob/v0.15.0/examples/nlp_example.py>`__
from the `accelerate GitHub
repo <https://github.com/huggingface/accelerate>`__.

.. code:: ipython3

    git_package = rh.git_package(git_url='https://github.com/huggingface/accelerate.git',
                                install_method='pip',
                                revision='v0.18.0')
    gpu.install_packages([git_package])


.. parsed-literal::
    :class: code-output

    INFO | 2023-06-08 19:57:11,991 | Installing packages on cluster rh-a10x: ['GitPackage: https://github.com/huggingface/accelerate.git@v0.18.0']


Additionally install any other necessary requirements to run the script.

.. code:: ipython3

    reqs = ['evaluate', 'transformers', 'datasets==2.3.2', 'scipy', 'scikit-learn', 'tqdm', 'tensorboard', 'torch==1.12.0']

    env = rh.env(reqs=reqs)
    env.to(gpu)

    # or
    # gpu.install_packages(reqs)

Now that we have the script and dependencies on the cluster, we can run
the script using ``gpu.run([command])``

.. code:: ipython3

    gpu.run(['python accelerate/examples/nlp_example.py'])

Terminate Cluster
-----------------

To terminate the cluster after you’re done using it, you can either use
the ``sky down cluster-name`` CLI command or ``cluster_var.down()``
Python API.

If you set up autostop for the cluster or in your configs (default to 30
min), the cluster will automatically terminate after that period of
inactivity.

.. code:: ipython3

    # cli
    !sky down rh-a10x

    # python
    # gpu.down()


.. parsed-literal::
    :class: code-output

    Terminating 1 cluster: rh-a10x. Proceed? [Y/n]: y
    [2K[1;36mTerminating 1 cluster[0m [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m  0%[0m [36m-:--:--[0m
    [2K[1;36mTerminating 1 cluster[0m [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m  0%[0m [36m-:--:--[0m
    [1A[2K[32mTerminating cluster rh-a10x...done.[0m
    [2K[1;36mTerminating 1 cluster[0m [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [35m100%[0m [36m0:00:00[0m
    [?25h[0m
