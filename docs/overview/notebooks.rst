Notebooks
====================================

If you prefer to work or debug in notebooks, you can call the following to tunnel a JupyterLab server into your local
browser from your Runhouse cluster or function:

.. code-block:: console

    $ runhouse notebook my_cluster

or in Python:

.. code-block:: python

    my_cluster.notebook()

If you'd like to use a hosted notebook service like Colab, you'll benefit a lot from creating a
Runhouse account to store your secrets and loading them into Colab with :code:`rh.login()`.
This is not required, and you can still drop them into the Colab VM manually.


Notes on notebooks
~~~~~~~~~~~~~~~~~~~
Notebooks are funny beasts. The code and variable inside them are not designed to be reused to shuttled around. As such:

1. If you want to :code:`rh.function` a function defined inside the notebook, it cannot contain variables or imports from outside the function, and you should assign a :code:`name` to the function. We will write the function out to a separate :code:`.py` file and import it from there, and the filename will be set to the :code:`function.name`.
2. If you really want to use local variables or avoid writing out the function, you can set :code:`serialize_notebook_fn=True` in :code:`rh.function()`. This will cloudpickle the function before sending it, but we do not support saving and reloading these kinds of functions (cloudpickle does not support this kind of reuse and it will create issues).
3. It is nearly always better to try to write your code in a .py file somewhere and import it into the notebook, rather than define important functions in the notebook itself. You can also use the :code:`%%writefile` magic to write your code into a file, and then import it back into the notebook.



If you want to sync down your code or data to local from the cluster afterwards:

.. code-block:: python

    rh.folder(path='remote_directory', system=rh.cluster('my_cluster').to('here', path='local_directory')
