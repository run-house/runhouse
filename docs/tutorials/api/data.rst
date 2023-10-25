Data: Folders, Tables, & Blobs
==============================

.. raw:: html

    <p><a href="https://colab.research.google.com/github/run-house/runhouse/blob/stable/docs/notebooks/api/data.ipynb">
    <img height="20px" width="117px" src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a></p>


Runhouse has several abstractions to provide a simple interface for
storing, recalling, and moving data between the user’s laptop, remote
compute, cloud storage, and specialized storage (e.g. data warehouses).

The Folder, Table, and Blob APIs provide least-common-denominator APIs
across providers, allowing users to easily specify the actions they want
to take on the data without needed to dig into provider-specific APIs.

Install Runhouse and Setup Cluster
----------------------------------

.. code:: ipython3

    !pip install "runhouse[aws]"

.. code:: ipython3

    import runhouse as rh

Optionally, login to Runhouse to sync credentials.

.. code:: ipython3

    !runhouse login

We also construct a Runhouse cluster object that we will use throughout
the tutorial. We won’t go in depth about clusters in this tutorial, but
you can refer to Getting Started for setup instructions, or the Compute
API tutorial for a more in-depth walkthrough of clusters.

.. code:: ipython3

    cluster = rh.ondemand_cluster(
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

.. code:: ipython3

    import os
    folder_name = "sample_folder"
    os.makedirs(folder_name, exist_ok=True)

    for i in range(5):
      with open(f'{folder_name}/{i}.txt', 'w') as f:
          f.write('i')

    local_path = f"{os.getcwd()}/{folder_name}"

To create a folder object, use the ``rh.folder()`` factory function, and
use ``.to()`` to send the folder to a remote cluster.

.. code:: ipython3

    local_folder = rh.folder(path=f"{os.getcwd()}/{folder_name}")
    cluster_folder = local_folder.to(system=cluster, path=folder_name)

    cluster.run([f"ls {folder_name}"])


.. parsed-literal::
    :class: code-output

    INFO | 2023-08-29 19:45:52.597164 | Copying folder from file:///Users/caroline/Documents/runhouse/runhouse/docs/notebooks/basics/sample_folder to: cpu-cluster, with path: sample_folder
    INFO | 2023-08-29 19:45:54.633598 | Running command on cpu-cluster: ls sample_folder


.. parsed-literal::
    :class: code-output

    0.txt
    1.txt
    2.txt
    3.txt
    4.txt




.. parsed-literal::
    :class: code-output

    [(0, '0.txt\n1.txt\n2.txt\n3.txt\n4.txt\n', '')]



You can also send the folder to file storage, such as S3, GS, and Azure.

.. code:: ipython3

    s3_folder = local_folder.to(system="s3")
    s3_folder.ls(full_paths=False)


.. parsed-literal::
    :class: code-output

    INFO | 2023-08-29 19:47:47.618511 | Copying folder from file:///Users/caroline/Documents/runhouse/runhouse/docs/notebooks/basics/sample_folder to: s3, with path: /runhouse-folder/a6f195296945409da432b2981f984ae7
    INFO | 2023-08-29 19:47:47.721743 | Found credentials in shared credentials file: ~/.aws/credentials
    INFO | 2023-08-29 19:47:48.796181 | Found credentials in shared credentials file: ~/.aws/credentials




.. parsed-literal::
    :class: code-output

    ['0.txt', '1.txt', '2.txt', '3.txt', '4.txt']



Similarly, you can send folders from a cluster to file storage, cluster
to cluster, or file storage to file storage. These are all done without
bouncing the folder off local.

.. code:: ipython3

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

Let’s step through an example using a Pandas table we upload to our s3
bucket using Runhouse.

.. code:: ipython3

    import pandas as pd
    df = pd.DataFrame(
            {"id": [1, 2, 3, 4, 5, 6], "grade": ["a", "b", "b", "a", "a", "e"]}
        )

    table_name = "sample_table"
    path = "/runhouse-table/sample_table"
    rh_table = rh.table(
        data=df, name=table_name, path=path, system="s3", mkdir=True
    ).write().save()


.. parsed-literal::
    :class: code-output

    INFO | 2023-08-29 19:55:29.834000 | Found credentials in shared credentials file: ~/.aws/credentials


.. code:: ipython3

    rh_table.data



.. parsed-literal::
    :class: code-output

     id grade
      1     a
      2     b
      3     b
      4     a
      5     a
      6     e



To sync over and save the table to a remote cluster, or to local
(“here”):

.. code:: ipython3

    cluster_table = rh_table.to(cluster)


.. parsed-literal::
    :class: code-output

    INFO | 2023-08-29 19:59:39.456856 | Copying folder from s3://runhouse-table/sample_table to: cpu-cluster, with path: ~/.cache/runhouse/82d19ef56425409fb92e5d4dfcd389e2
    INFO | 2023-08-29 19:59:39.458405 | Running command on cpu-cluster: aws --version >/dev/null 2>&1 || pip3 install awscli && aws s3 sync --no-follow-symlinks s3://runhouse-table/sample_table ~/.cache/runhouse/82d19ef56425409fb92e5d4dfcd389e2


.. parsed-literal::
    :class: code-output

    download: s3://runhouse-table/sample_table/d68a64f755014c049b6e97b120db5d0f.parquet to .cache/runhouse/82d19ef56425409fb92e5d4dfcd389e2/d68a64f755014c049b6e97b120db5d0f.parquet
    download: s3://runhouse-table/sample_table/ebf7bbc1b22e4172b162b723b4b234f2.parquet to .cache/runhouse/82d19ef56425409fb92e5d4dfcd389e2/ebf7bbc1b22e4172b162b723b4b234f2.parquet
    download: s3://runhouse-table/sample_table/53d00aa5fa2148dd9f4d9836f7b6a9be.parquet to .cache/runhouse/82d19ef56425409fb92e5d4dfcd389e2/53d00aa5fa2148dd9f4d9836f7b6a9be.parquet
    download: s3://runhouse-table/sample_table/2d0aed0ba49d42509ae9124368a74323.parquet to .cache/runhouse/82d19ef56425409fb92e5d4dfcd389e2/2d0aed0ba49d42509ae9124368a74323.parquet
    download: s3://runhouse-table/sample_table/ea3841db70874ee7aade6ff1299325c5.parquet to .cache/runhouse/82d19ef56425409fb92e5d4dfcd389e2/ea3841db70874ee7aade6ff1299325c5.parquet
    download: s3://runhouse-table/sample_table/e7a7dce218054b6aa2b0853c12afe952.parquet to .cache/runhouse/82d19ef56425409fb92e5d4dfcd389e2/e7a7dce218054b6aa2b0853c12afe952.parquet


.. code:: ipython3

    local_table = rh_table.to('here')


.. parsed-literal::
    :class: code-output

    INFO | 2023-08-29 19:59:49.336813 | Copying folder from s3://runhouse-table/sample_table to: file, with path: /Users/caroline/Documents/runhouse/runhouse/docs/notebooks/basics/sample_table


To stream batches of the table, we reload the table object, but with an
iterable ``.data`` field, using the ``rh.table`` constructor and passing
in the name.

Note that you can’t directly do this with the original table object, as
its ``.data`` field is the original ``data`` passed in, and not
necessarily in an iterable format.

.. code:: ipython3

    reloaded_table = rh.table(name=table_name)

.. code:: ipython3

    batches = reloaded_table.stream(batch_size=2)
    for batch in batches:
        print(batch)


.. parsed-literal::
    :class: code-output

       id grade
    0   1     a
    1   2     b
       id grade
    0   3     b
    1   4     a
       id grade
    0   5     a
    1   6     e



Blobs
-----

The Runhouse Blob API represents an entity for storing arbitrary data.
Blobs are associated with a system (local, remote, or file storage), and
can be written down or synced to systems.

.. code:: ipython3

    import json
    import pickle

    blob_data = list(range(50))

.. code:: ipython3

    # create and save local blob
    local_blob = rh.blob(
            name="local_blob",
            data=blob_data,
            system="file",
            path="local_blob.pickle"
        ).save()


.. parsed-literal::
    :class: code-output

    INFO | 2023-08-29 20:57:10.570715 | Creating new file folder if it does not already exist in path: /Users/caroline/Documents/runhouse/runhouse


.. code:: ipython3

    # to sync the blob to remote or fs
    local_blob.to(system=cluster)
    local_blob.to(system="s3")

.. code:: ipython3

    # create blob on s3
    rh.blob(
        data=blob_data,
        system="s3",
        path=f"/runhouse-blob/sample_blob.pickle",
    )

    # create blob on cluster
    rh.blob(
        data=blob_data,
        system=cluster,
    )

To get the contents from a blob, use ``.fetch()``:

.. code:: ipython3

    print(local_blob.fetch())




.. parsed-literal::
    :class: code-output

    '[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49]'



Now that you understand the basics, feel free to play around with more
complicated scenarios! You can also check out our additional API and
example usage tutorials on our `docs
site <https://www.run.house/docs>`__.

Cluster Termination
-------------------

.. code:: ipython3

    !sky down cpu-cluster
    # or
    cluster.teardown()

.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #008000; text-decoration-color: #008000">⠹</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">Terminating </span><span style="color: #008000; text-decoration-color: #008000; font-weight: bold">cpu-cluster</span>
    </pre>
