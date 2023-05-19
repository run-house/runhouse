Data: Folders, Blobs, Tables
============================

.. raw:: html

    <p><a href="https://colab.research.google.com/github/run-house/runhouse/blob/stable/docs/notebooks/data.ipynb">
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

    INFO | 2023-05-08 20:15:23,316 | Creating new file folder if it does not already exist in path: /content/sample_folder
    INFO | 2023-05-08 20:15:23,318 | Copying folder from file:///content/sample_folder to: cpu-cluster, with path: sample_folder
    INFO | 2023-05-08 20:15:24,766 | Running command on cpu-cluster: ls sample_folder
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

    INFO | 2023-05-08 21:49:13,620 | Attempting to load config for /carolineechen/file from RNS.
    INFO | 2023-05-08 21:49:13,684 | No config found in RNS: {'detail': 'Resource does not exist'}
    INFO | 2023-05-08 21:49:13,687 | Creating new file folder if it does not already exist in path: /root/.cache/runhouse/tables/0b9b0c0c5afc4d03b475db6ec61f7b7b
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

To stream batches of the table, we can create a new table object with an
iterable ``.data`` field using the ``rh.table`` constructor and passing
in the name.

.. code:: python

    reloaded_table = rh.table(name=table_name)
    batches = reloaded_table.stream(batch_size=2)
    for _, batch in batches:
        print(batch)

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

    INFO | 2023-05-08 21:40:29,141 | Attempting to load config for /carolineechen/local_blob from RNS.
    INFO | 2023-05-08 21:40:29,212 | Attempting to load config for /carolineechen/file from RNS.
    INFO | 2023-05-08 21:40:29,267 | No config found in RNS: {'detail': 'Resource does not exist'}
    INFO | 2023-05-08 21:40:29,269 | Creating new file folder if it does not already exist in path: /root/.cache/runhouse/blobs/aa9001761bb14d13bd3545b1f6127a6e/carolineechen
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49]
    INFO | 2023-05-08 21:40:29,274 | Attempting to load config for /carolineechen/local_blob from RNS.
    INFO | 2023-05-08 21:40:29,332 | Attempting to load config for /carolineechen/file from RNS.
    INFO | 2023-05-08 21:40:29,388 | No config found in RNS: {'detail': 'Resource does not exist'}
    INFO | 2023-05-08 21:40:29,390 | Creating new file folder if it does not already exist in path: /root/.cache/runhouse/blobs/aa9001761bb14d13bd3545b1f6127a6e/carolineechen
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49]


.. code:: python

    # create blob on s3
    rh.blob(data=blob_data, system="s3").write()

    # create blob from cluster
    rh.blob(path="path/on/cluster", system=cluster)

Terminate Cluster
-----------------

.. code:: python

    !sky down cpu-cluster
    # or
    cluster.teardown()
