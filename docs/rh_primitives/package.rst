Package
====================================
A Package is a Runhouse primitive for sharing code between various systems (ex: s3, cluster, local).


Package Factory Method
~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: runhouse.package


Package Class
~~~~~~~~~~~~~

.. autoclass:: runhouse.Package
   :members:
   :exclude-members:

    .. automethod:: __init__


GitPackage Docs
~~~~~~~~~~~~~~~

.. autoclass:: runhouse.GitPackage
   :members:
   :exclude-members:

    .. automethod:: __init__

API Usage
~~~~~~~~~

Packages are often times installed onto clusters using ``my_cluster.install_packages()``, or by setting the
``reqs`` of a Cluster or Function. There are currently four package install methods:

- ``local``: Install packages to a Folder or a given path to a directory.
- ``reqs``: Install a ``requirements.txt`` file from the working directory.
- ``pip``: Runs ``pip install`` for the provided packages.
- ``conda``: Runs ``conda install`` for the provided packages.

.. code:: python

   my_cluster(name,
              reqs=['local:./',          # local
                    'requirements.txt',  # reqs
                    'pip:diffusers',     # pip
                    'conda:pytorh',    # conda
              ])

To install a Git package using just the GitHub URL (and optionally, the revision), without needing
to clone or copy down the code yourself:

.. code:: python

   rh.GitPackage(git_url='https://github.com/huggingface/diffusers.git',
                 install_method='pip',
                 revision='v0.11.1')
