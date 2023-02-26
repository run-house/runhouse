Data
=======================================

The :ref:`Folder`, :ref:`Table`, and :ref:`Blob` APIs provide a simple interface for storing, recalling, and
moving data between the user's laptop, remote compute, cloud storage, and specialized storage (e.g. data warehouses).
They provide least-common-denominator APIs across providers, allowing users to easily specify the actions they want
to take on the data without needed to dig into provider-specific APIs.

For each of these data primitives, Runhouse provides APIs which make the data highly accessible regardless of
where it lives. While the data itself is saved to a particular system (ex: local, cluster, s3, gs),
you can still access the data from any other system.

Some common use cases include streaming data from your cluster to your laptop, mindlessly copying your model
checkpoints generated on that cluster directly to s3 without having to bounce that data back to your laptop, or copying
an existing folder from your laptop to a cluster or s3 bucket.

We'd like to extend this to other data concepts in the future, like kv-stores, time-series, vector and graph databases, etc.

Folders
-------
A :ref:`Folder` is useful for managing the various Runhouse resources created within organizations and teams.
It provides a reasonable option for teams to have shared resources without having to do any explicit coordination
through a separate source of truth. A Folder's contents are both what it physically contains and what it
symbolically contains in the object.

For example, let's say you generated a :ref:`Table` containing a tokenized dataset and saved it to your team's
folder of shared resources (in path: :code:`/ds/bert_preproc`.):

.. code-block:: python

    my_table = rh.table(
        data=tokenized_dataset,
        name="@/bert_tokenized",
        path="/ds/bert_preproc",
        system="s3",
        mkdir=True,
    ).save()

Runhouse will then save the underlying dataset to that particular bucket in your specified system (ex: s3)
within the :code:`bert_preproc` directory, which can be then be accessed by the rest of your team.

We currently support a variety of systems where a folder can live:

- :code:`file`: In a Local file system.
- :code:`github`: On GitHub (based on provided URL).
- :code:`sftp` / :code:`ssh`: On a :ref:`Cluster`.
- :code:`s3`: Bucket in AWS.
- :code:`gs`: Bucket in GCS.


Advanced Folder Usage
~~~~~~~~~~~~~~~~~~~~~
Let's demonstrate how you can easily copy a folder from one system to another. In this example we'll
copy a local folder to a cluster.

.. code-block:: python

    local_folder = rh.folder(Path.cwd(), name='my_local_folder')

    # Use a Runhouse builtin cluster, make sure the cluster is up if it isn't already
    c = rh.cluster("^rh-cpu").up_if_not()

    # `from_cluster` will create a remote folder from our path on the cluster
    cluster_folder = local_folder.to(system=c).from_cluster(c)

    # Confirm that the folder contents are now saved on the `rh-cpu` cluster
    print(cluster_folder.ls(full_paths=False))


We can just as easily send this local folder to a cloud storage system, such as s3:

.. code-block:: python

    s3_folder = rh.folder(name='my_local_folder').to(system="s3")


Tables
------
Runhouse supports a variety of different :ref:`Table` types based on the table's underlying data type.
By default we store tables as parquet, but Runhouse provides a number of Table subclass implementations with
convenient APIs for writing, partitioning, fetching and streaming the data:

- :code:`Table`: Base table implementation. Supports any data type that can be written to parquet (ex: `pyArrow <https://arrow.apache.org/docs/python/generated/pyarrow.Table.html>`_).
- :code:`RayTable`: `Ray Datasets <https://docs.ray.io/en/latest/data/api/dataset.html#ray.data.Dataset>`_
- :code:`HuggingFaceTable`: `HuggingFace Datasets <https://huggingface.co/docs/datasets/index>`_
- :code:`PandasTable`: `Pandas DataFrames <https://pandas.pydata.org/docs/reference/frame.html>`_
- :code:`DaskTable`: `Dask DataFrames <https://docs.dask.org/en/stable/dataframe.html>`_

.. note::
    In the near term, we plan on supporting Spark, Rapids, and BigQuery. Please let us know if there is a
    particular Table abstraction that would be useful to you.


Advanced Table Usage
~~~~~~~~~~~~~~~~~~~~
Let's demonstrate how we can easily create a Table backed by a Pandas DataFrame that lives in s3,
and access that data from any other system:

.. code-block:: python

    data = pd.DataFrame(...)
    my_table = rh.table(
        data=data,
        name="@/my_pandas_table",
        path=f"/preproc-data/pandas", # path to s3 folder where the table will live
        system="s3",
        mkdir=True,
    ).save()


Now we can easily stream this table from our laptop, an existing cluster, a notebook, etc:

.. code-block:: python

    reloaded_table = rh.table(name="my_pandas_table")

This :code:`reloaded_table` holds a reference to the table's path.

.. code-block:: python

    batches = reloaded_table.stream(batch_size=100)
        for batch in batches:
            ....

Our `BERT Pipeline Preprocessing Tutorial <https://github.com/run-house/tutorials/blob/main/t05_BERT_pipeline/p01_preprocess.py>`_
showcases the accessibility and portability that a Table can provide. We create a tokenized dataset Table object on a
cluster, then stream that data in directly from the cluster.

Blobs
-----
A :ref:`Blob` represents a single serialized file stored in a particular system.
Blobs are useful for dropping data into storage without worrying about exactly where it sits, with Runhouse
handling saving down and retrieving the Blob for you.

For example, if you want to save a model checkpoint for future reuse, use the Blob interface
to easily save it in your desired cloud storage system.

Our `BERT Pipeline Fine-Tuning Tutorial <https://github.com/run-house/tutorials/blob/main/t05_BERT_pipeline/p02_fine_tune.py/>`_
shows how we can use a Blob to save a trained BERT fine tuning model locally on a cluster.
When finished, we can send the Blob from the cluster directly to an s3 bucket for persistence.

Please note Runhouse does not make any assumptions about deserializing the underlying blob data.
In this example we load an existing blob and deserialize ourselves with :code:`pickle`:

.. code-block:: python

    my_blob = Blob.from_name("my_blob")
    raw_data = my_blob.fetch()
    # need to do the deserialization ourselves
    res = pickle.loads(raw_data)
