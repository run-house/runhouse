Data
=======================================

The :ref:`Folder`, :ref:`Table`, and :ref:`Blob` APIs provide a simple interface for storing, recalling, and
moving data between the user's laptop, remote compute, cloud storage, and specialized storage (e.g. data warehouses).
They provide least-common-denominator APIs across providers, allowing users to easily specify the actions they want
to take on the data without needed to dig into provider-specific APIs.

For each of these data primitives, Runhouse provides APIs the data highly accessible, regardless of
where the data lives. While the data itself is saved down in a particular system (ex: local, cluster, s3, gs),
you can still access the data from any other system.

For example, you can easily stream data from your 32 CPU cluster to your laptop, or mindlessly copy your model
checkpoints generated on that cluster directly to S3, without having to bounce that data back to your laptop.

We'd like to extend this to other data concepts in the future, like kv-stores, time-series, vector and graph databases, etc.

Folders
=======================================
Folders are useful for managing the various Runhouse resources within organizations and teams.
It provides a reasonable option for teams to have shared resources without having to do any explicit coordination
through a separate source of truth. A Folder's contents are both what it physically contains and what it
symbolically contains in the object.

For example, you may create a Table within the Folder managed by the Data Science team called :code:`/ds/bert_preproc`.
This will save the underlying data in your specified folder system (ex: :code:`s3`) to the :code:`bert_preproc` directory
in the :code:`ds` bucket.

We currently support a variety of systems where a :ref:`Folder` can live:

- :code:`file`: Local file system.
- :code:`github`: Interact with a git URL.
- :code:`sftp` / :code:`ssh`: Interact with a folder on a cluster.
- :code:`s3`: Interact with a bucket on S3.
- :code:`gs`. Interact with a bucket on GCS.


Advanced Folder Usage
~~~~~~~~~~~~~~~~~~~~~
Let's show how you can easily copy a folder from one system to another. In this case we'll
demonstrate copying a local folder of .txt files to a cluster.

.. code-block:: python

    local_folder = rh.folder(path=Path.cwd() / "folder_tmp")
    local_folder.mkdir()

    # Add some txt files to the local folder
    local_folder.put({f"sample_file_{i}.txt": f"file{i}".encode() for i in range(3)})


Now that we have our local folder with some files, let's copy them to the :code:`rh-cpu` cluster:

.. code-block:: python

    # Use a Runhouse builtin cluster, make sure the cluster is up if it isn't already
    c = rh.cluster("^rh-cpu").up_if_not()

    # `from_cluster` will create a remote folder from our path on the cluster
    cluster_folder = local_folder.to(system=c).from_cluster(c)

    # Show that these .txt files are now located on the rh-cpu cluster
    print(cluster_folder.ls(full_paths=False))


Tables
=======================================
We currently support a variety of different table types based on your desired underlying infra. By default we store
tables as parquet in blob storage, but Runhouse provides a number of Table subclass implementations with
:ref:`convenient APIs <Table>` for writing, partitioning, fetching, and streaming the underlying data:

- :code:`Table`: Base table implementation. Supports any data type that can be written to parquet (ex: pyArrow).
- :code:`RayTable`: Ray Datasets.
- :code:`HuggingFaceTable`: HuggingFace Datasets.
- :code:`PandasTable`: Pandas Dataframes.
- :code:`DaskTable`: Dask Dataframes.

.. note::
    In the near term, we plan on supporting Spark, Rapids, and BigQuery. Please let us know if there is a
    particular Table abstraction that would be useful to you.


Advanced Table Usage
~~~~~~~~~~~~~~~~~~~~

Let's demonstrate how we can easily create a Pandas dataframe table that lives in s3, and access
that data from any other system:

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

    reloaded_table = rh.table(name="@/my_test_fetch_dask_table", dryrun=True)

This :code:`reloaded_table` holds a reference to the table's path.

.. code-block:: python

    batches = reloaded_table.stream(batch_size=100)
        for batch in batches:
            ....

Our `BERT Pipeline Preprocessing Tutorial <https://github.com/run-house/tutorials/blob/main/t05_BERT_pipeline/p01_preprocess.py>`_
showcases the accessibility and portability that a Table can provide. We create a tokenized dataset Table object on a
cluster, then stream that data in directly from the cluster later on.

Blobs
=======================================
A :ref:`Blob` represents a single serialized file stored in a particular system.
Blobs are useful for dropping data into storage without worrying about exactly where it sits, with Runhouse
handling saving down and retrieving the Blob for you.

For example, if you want to save a model checkpoint for future reuse, use the Blob interface
to easily save it in your desired system.

Our `BERT Pipeline Fine-Tuning Tutorial <https://github.com/run-house/tutorials/blob/main/t05_BERT_pipeline/p02_fine_tune.py/>`_
shows how we can use a Blob to save a trained BERT fine tuning model locally on a cluster.
When finished, we can send the Blob from the cluster directly to an s3 bucket for persistence.
