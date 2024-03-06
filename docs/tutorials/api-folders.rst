Folders
=======

.. raw:: html

    <p><a href="https://colab.research.google.com/github/run-house/notebooks/blob/stable/docs/api-folder.ipynb">
    <img height="20px" width="117px" src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a></p>

The Runhouse Folder makes it easy to send folders and files between your
local environment, a cluster, or your cloud storage (using your own
credentials), without needing to learn and provider-specific APIs.

Installation and Setup
----------------------

To install base runhouse:

.. code:: ipython3

    !pip install runhouse

Runhouse supports sending folders to/from cloud storage such as s3, gcs,
azure. To download provider-specific libraries that are used under the
hood, you can install ``"runhouse[aws/gcp/azure]"``. In this tutorial we
demonstrate with s3 and gcs, and install ``"runhouse[aws, gcp]"``.

.. code:: ipython3

    !pip install "runhouse[aws, gcp]"

If you would like to use ``s3`` or ``gcs``, please make sure to also set
up your credentials locally. You can see the instructions for this by
running ``sky check``.

.. code:: ipython3

    import runhouse as rh

Folder Setup
~~~~~~~~~~~~

Here we define a simple folder structure in our current directory, a
simple ``sample-folder`` consisting of 5 files, ``1-5.txt``.

.. code:: ipython3

    import os
    folder_name = "sample-folder"
    os.makedirs(folder_name, exist_ok=True)

    for i in range(5):
      with open(f'{folder_name}/{i}.txt', 'w') as f:
          f.write('i')

    local_path = f"{os.getcwd()}/{folder_name}"
    local_path




.. parsed-literal::
    :class: code-output

    '/Users/caroline/Documents/runhouse/notebooks/docs/sample-folder'



Cluster Setup
~~~~~~~~~~~~~

Launch a basic cluster, as the tutorial will demonstrate sending the
local folder to the cluster. You can learn more about clusters in the
`Cluster
tutorial <https://www.run.house/docs/tutorials/api-clusters>`__.

.. code:: ipython3

    cluster = rh.cluster(
        name="rh-cluster",
        instance_type="CPU:2+",
        provider="aws",
    )
    cluster.up_if_not()

Runhouse Folder
---------------

Construct a Runhouse folder object with ``rh.folder``, passing in the
path of the folder you’d like it to represent. Optionally pass in a
``system=<cluster>/s3/gcs/azure`` that the folder lives on.

Here, we construct a Runhouse folder object that represents the
``sample-folder`` that we created earlier.

.. code:: ipython3

    local_folder = rh.folder(path=local_path)

To print the full paths, call ``.ls()``, or for relative paths, call
``.ls(full_paths=False)``.

.. code:: ipython3

    local_folder.ls(full_paths=False)




.. parsed-literal::
    :class: code-output

    ['4.txt', '3.txt', '2.txt', '0.txt', '1.txt']



To: Cluster
~~~~~~~~~~~

To send it to a cluster, call ``.to(system=cluster)``, and optionally
pass in a path. If no path is provided, it will be automatically
generated. The path can be retrieved by calling ``.path`` on the
resulting object.

.. code:: ipython3

    cluster_folder = local_folder.to(system=cluster, path=folder_name)


.. parsed-literal::
    :class: code-output

    INFO | 2024-03-06 04:35:08.517625 | Copying folder from file:///Users/caroline/Documents/runhouse/notebooks/docs/sample-folder to: rh-cluster, with path: sample-folder


.. code:: ipython3

    cluster_folder.ls()




.. parsed-literal::
    :class: code-output

    ['sample-folder/3.txt',
     'sample-folder/0.txt',
     'sample-folder/4.txt',
     'sample-folder/2.txt',
     'sample-folder/1.txt']



.. code:: ipython3

    cluster_folder.path




.. parsed-literal::
    :class: code-output

    'sample-folder'



To: S3/GCS
~~~~~~~~~~

Sending to S3/GCS is similar, call ``.to(system=s3/gcs)``.

.. code:: ipython3

    gs_folder = local_folder.to(system="gs")


.. parsed-literal::
    :class: code-output

    INFO | 2024-03-06 04:35:38.607986 | Copying folder from file:///Users/caroline/Documents/runhouse/notebooks/docs/sample-folder to: gs, with path: /runhouse-folder/bd489bb276734f7f8c23e401e6bb2b51


.. code:: ipython3

    gs_folder.ls(full_paths=False)




.. parsed-literal::
    :class: code-output

    ['0.txt', '1.txt', '2.txt', '3.txt', '4.txt']



Similarly, for s3:

.. code:: ipython3

    s3_folder = local_folder.to(system="s3")


.. parsed-literal::
    :class: code-output

    INFO | 2024-03-06 04:36:04.390441 | Copying folder from file:///Users/caroline/Documents/runhouse/notebooks/docs/sample-folder to: s3, with path: /runhouse-folder/dae8c16b71a744cb976da0dace7c4db2


.. code:: ipython3

    s3_folder.ls(full_paths=False)




.. parsed-literal::
    :class: code-output

    ['0.txt', '1.txt', '2.txt', '3.txt', '4.txt']



To: Here
~~~~~~~~

The keyword for sending to local is ``.to("here")``.

.. code:: ipython3

    new_local_folder = s3_folder.to("here", path="new-sample-folder")


.. parsed-literal::
    :class: code-output

    INFO | 2024-03-06 04:38:01.269441 | Copying folder from s3://runhouse-folder/dae8c16b71a744cb976da0dace7c4db2 to: file, with path: new-sample-folder


.. code:: ipython3

    new_local_folder.ls(full_paths=False)




.. parsed-literal::
    :class: code-output

    ['4.txt', '3.txt', '2.txt', '0.txt', '1.txt']



And more..
~~~~~~~~~~

Folders can be sent between any pair of local, cluster, or cloud
storage, including between different clusters, or within the same cloud
storage but duplicating the folder to a second location in storage.
