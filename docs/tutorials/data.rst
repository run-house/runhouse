Data: Folders, Blobs, Tables
============================

.. raw:: html

    <p><a href="https://colab.research.google.com/github/run-house/runhouse/blob/main/docs/notebooks/data.ipynb">
    <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a></p>


Runhouse has several abstractions to provide a simple interface for
storing, recalling, and moving data between the user’s laptop, remote
compute, cloud storage, and specialized storage (e.g. data warehouses).

The Folder, Table, and Blob APIs provide least-common-denominator APIs
across providers, allowing users to easily specify the actions they want
to take on the data without needed to dig into provider-specific APIs.

Install Runhouse and Setup Cluster
----------------------------------

.. code:: python

    !pip install runhouse[aws]

.. code:: python

    import runhouse as rh


Optionally, login to Runhouse to sync credentials.

.. code:: python

    !runhouse login

We also construct a Runhouse cluster object that we will use throughout
the tutorial. We won’t go in depth about clusters in this tutorial, but
you can refer to Getting Started for setup instructions, or the Compute
API tutorial for a more in-depth walkthrough of clusters.

.. code:: python

    cluster = rh.cluster(
                  name="cpu-cluster",
                  instance_type="CPU:8",
                  provider="cheapest",       # "AWS", "GCP", "Azure", "Lambda", or "cheapest" (default)
                  autostop_mins=60,          # Optional, defaults to default_autostop_mins; -1 suspends autostop
              )
    cluster.up()

Folders
-------

The Runhouse Folder API allows for creating references to folders, and
syncing them between local, remote clusters, or file storage (S3, GS,
Azure).

Let’s construct a sample dummy folder locally, that we’ll use to
demonstrate.

.. code:: python

    import os
    folder_name = "sample_folder"
    os.makedirs(folder_name, exist_ok=True)

    for i in range(5):
      with open(f'{folder_name}/{i}.txt', 'w') as f:
          f.write('i')

    local_path = f"{os.getcwd()}/{folder_name}"

To create a folder object, use the ``rh.folder()`` factory function, and
use ``.to()`` to send the folder to a remote cluster.

.. code:: python

    local_folder = rh.folder(path=f"{os.getcwd()}/{folder_name}")
    cluster_folder = local_folder.to(system=cluster, path=folder_name)

    cluster.run([f"ls {folder_name}"])


.. parsed-literal::

    INFO | 2023-05-18 22:04:19,262 | Creating new file folder if it does not already exist in path: /content/sample_folder
    INFO | 2023-05-18 22:04:19,270 | Copying folder from file:///content/sample_folder to: cpu-cluster, with path: sample_folder
    INFO | 2023-05-18 22:04:21,170 | Running command on cpu-cluster: ls sample_folder
    0.txt
    1.txt
    2.txt
    3.txt
    4.txt




.. parsed-literal::

    [(0, '0.txt\n1.txt\n2.txt\n3.txt\n4.txt\n', '')]



You can also send the folder to file storage, such as S3, GS, and Azure.

.. code:: python

    s3_folder = local_folder.to(system="s3")
    s3_folder.ls(full_paths=False)


.. parsed-literal::

    INFO | 2023-05-18 22:04:25,030 | Copying folder from file:///content/sample_folder to: s3, with path: /runhouse-folder/79fe2eef03744148852156a003445885
    INFO | 2023-05-18 22:04:25,034 | Attempting to load config for /carolineechen/s3 from RNS.
    INFO | 2023-05-18 22:04:25,275 | No config found in RNS: {'detail': 'Resource does not exist'}
    INFO | 2023-05-18 22:04:26,717 | Found credentials in shared credentials file: ~/.aws/credentials




.. parsed-literal::

    ['0.txt', '1.txt', '2.txt', '3.txt', '4.txt']



Similarly, you can send folders from a cluster to file storage, cluster
to cluster, or file storage to file storage. These are all done without
bouncing the folder off local.

.. code:: python

    cluster_folder.to(system=another_cluster)  # cluster to cluster
    cluster_folder.to(system="s3")             # cluster to fs
    s3_folder.to(system=cluster)               # fs to cluster
    s3_folder.to(system="gs")                  # fs to fs

Tables
------

The Runhouse Table API allows for abstracting tabular data storage, and
supports interfaces for HuggingFace, Dask, Pandas, Rapids, and Ray
tables (more in progress!).

These can be synced and written down to local, remote clusters, or file
storage (S3, GS, Azure).

Let’s step through an example using Pandas tables:

.. code:: python

    import pandas as pd
    df = pd.DataFrame(
            {"id": [1, 2, 3, 4, 5, 6], "grade": ["a", "b", "b", "a", "a", "e"]}
        )

    table_name = "sample_table"
    rh_table = rh.table(data=df, name=table_name)
    print(rh_table.data)


.. parsed-literal::

    INFO | 2023-05-18 22:10:14,856 | Attempting to load config for /carolineechen/sample_table from RNS.
    INFO | 2023-05-18 22:10:15,076 | No config found in RNS: {'detail': 'Resource does not exist'}
    INFO | 2023-05-18 22:10:15,078 | Attempting to load config for /carolineechen/file from RNS.
    INFO | 2023-05-18 22:10:15,261 | No config found in RNS: {'detail': 'Resource does not exist'}
    INFO | 2023-05-18 22:10:15,266 | Creating new file folder if it does not already exist in path: /root/.cache/runhouse/tables/carolineechen/sample_table
       id grade
    0   1     a
    1   2     b
    2   3     b
    3   4     a
    4   5     a
    5   6     e


To sync over and save the table to file storage, like S3, or to a remote
cluster:

.. code:: python

    rh_table.to(system="s3")
    rh_table.to(cluster)


.. parsed-literal::

    INFO | 2023-05-18 22:10:24,487 | Copying folder from file:///root/.cache/runhouse/tables/carolineechen/sample_table to: s3, with path: /runhouse-folder/9215396cea4040c093997f3d5ae48943
    INFO | 2023-05-18 22:10:24,490 | Attempting to load config for /carolineechen/s3 from RNS.
    INFO | 2023-05-18 22:10:24,648 | No config found in RNS: {'detail': 'Resource does not exist'}
    INFO | 2023-05-18 22:10:25,468 | Copying folder from file:///root/.cache/runhouse/tables/carolineechen/sample_table to: cpu-cluster, with path: ~/.cache/runhouse/ed2fd40ca63140408444deca935528ec




.. parsed-literal::

    <runhouse.rns.tables.pandas_table.PandasTable at 0x7fc346de6c50>



To stream batches of the table, we reload the table object, but with an
iterable ``.data`` field, using the ``rh.table`` constructor and passing
in the name.

Note that you can’t directly do this with the original table object, as
it’s ``.data`` field is the original ``data`` passed in, and not
necessarily in an iterable format.

.. code:: python

    reloaded_table = rh.table(name=table_name)

.. code:: python

    batches = reloaded_table.stream(batch_size=2)
    for _, batch in batches:
        print(batch)


.. parsed-literal::

    2023-05-18 22:13:41,227	WARNING read_api.py:330 -- ⚠️  The number of blocks in this dataset (0) limits its parallelism to 0 concurrent tasks. This is much less than the number of available CPU slots in the cluster. Use `.repartition(n)` to increase the number of dataset blocks.
    Parquet Files Sample: : 0it [00:00, ?it/s]


Blobs
-----

The Runhouse Blob API represents an entity for storing arbitrary data.
Blobs are associated with a system (local, remote, or file storage), and
can be written down or synced to systems.

.. code:: python

    import json
    import pickle

    blob_data = pickle.dumps(json.dumps(list(range(50))))

.. code:: python

    # create local blob and write contents to file
    local_blob = rh.blob(name="local_blob", data=blob_data).write()
    print(pickle.loads(local_blob.data))

    # reload local blob
    reloaded_blob = rh.blob(name="local_blob")
    print(pickle.loads(reloaded_blob.fetch()))

    # to sync the blob to remote or fs
    local_blob.to(system=cluster)
    local_blob.to(system="s3")


.. parsed-literal::

    INFO | 2023-05-18 22:15:54,332 | Attempting to load config for /carolineechen/local_blob from RNS.
    INFO | 2023-05-18 22:15:54,524 | Attempting to load config for /carolineechen/file from RNS.
    INFO | 2023-05-18 22:15:54,690 | No config found in RNS: {'detail': 'Resource does not exist'}
    INFO | 2023-05-18 22:15:54,692 | Creating new file folder if it does not already exist in path: /root/.cache/runhouse/blobs/aa9001761bb14d13bd3545b1f6127a6e/carolineechen
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49]
    INFO | 2023-05-18 22:15:54,695 | Attempting to load config for /carolineechen/local_blob from RNS.
    INFO | 2023-05-18 22:15:54,854 | Attempting to load config for /carolineechen/file from RNS.
    INFO | 2023-05-18 22:15:55,015 | No config found in RNS: {'detail': 'Resource does not exist'}
    INFO | 2023-05-18 22:15:55,017 | Creating new file folder if it does not already exist in path: /root/.cache/runhouse/blobs/aa9001761bb14d13bd3545b1f6127a6e/carolineechen
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49]
    INFO | 2023-05-18 22:15:55,024 | Copying folder from file:///root/.cache/runhouse/blobs/aa9001761bb14d13bd3545b1f6127a6e/carolineechen to: cpu-cluster, with path: ~/.cache/runhouse/c4ff6d2954124f119932689ff9afa58b
    INFO | 2023-05-18 22:15:56,856 | Copying folder from file:///root/.cache/runhouse/blobs/aa9001761bb14d13bd3545b1f6127a6e/carolineechen to: s3, with path: /runhouse-folder/932d21850d2a43d8badd485e0d583758
    INFO | 2023-05-18 22:15:56,861 | Attempting to load config for /carolineechen/s3 from RNS.
    INFO | 2023-05-18 22:15:57,029 | No config found in RNS: {'detail': 'Resource does not exist'}




.. parsed-literal::

    <runhouse.rns.blob.Blob at 0x7fc33c268550>



.. code:: python

    # create blob on s3
    rh.blob(data=blob_data, system="s3").write()

    # create blob from cluster
    rh.blob(path="path/on/cluster", system=cluster)


.. parsed-literal::

    INFO | 2023-05-18 22:16:05,189 | Attempting to load config for /carolineechen/s3 from RNS.
    INFO | 2023-05-18 22:16:05,352 | No config found in RNS: {'detail': 'Resource does not exist'}
    INFO | 2023-05-18 22:16:05,354 | Creating new s3 folder if it does not already exist in path: /runhouse-blob/d135efb148b14ae9a05d50d0ba4c7c82
    INFO | 2023-05-18 22:16:05,374 | Found credentials in shared credentials file: ~/.aws/credentials
    INFO | 2023-05-18 22:16:05,863 | Creating new ssh folder if it does not already exist in path: path/on
    INFO | 2023-05-18 22:16:05,919 | Opening SSH connection to 3.93.183.178, port 22
    INFO | 2023-05-18 22:16:05,951 | [conn=0] Connected to SSH server at 3.93.183.178, port 22
    INFO | 2023-05-18 22:16:05,952 | [conn=0]   Local address: 172.28.0.12, port 44872
    INFO | 2023-05-18 22:16:05,955 | [conn=0]   Peer address: 3.93.183.178, port 22
    INFO | 2023-05-18 22:16:06,083 | [conn=0] Beginning auth for user ubuntu
    INFO | 2023-05-18 22:16:06,198 | [conn=0] Auth for user ubuntu succeeded
    INFO | 2023-05-18 22:16:06,201 | [conn=0, chan=0] Requesting new SSH session
    INFO | 2023-05-18 22:16:06,540 | [conn=0, chan=0]   Subsystem: sftp
    INFO | 2023-05-18 22:16:06,573 | [conn=0, chan=0] Starting SFTP client




.. parsed-literal::

    <runhouse.rns.blob.Blob at 0x7fc33c27a710>



To get the contents from a blob:

.. code:: python

    raw_data = local_blob.fetch()
    pickle.loads(raw_data)  # deserialization




.. parsed-literal::

    '[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49]'



Now that you understand the basics, feel free to play around with more
complicated scenarios! You can also check out our additional API and
example usage tutorials on our `docs
site <https://runhouse-docs.readthedocs-hosted.com/en/latest/index.html>`__.

Cluster Termination
-------------------

.. code:: python

    !sky down cpu-cluster
    # or
    cluster.teardown()

.. code:: python

    cluster.teardown()
