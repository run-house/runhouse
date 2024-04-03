Mapper
====================================
Mapper is a built-in Module for parallelizing a function or module method over a list of inputs. It
holds a pool of replicas of the function or module, distributes the inputs to the replicas, and collects
the results. The replicas are either created by the mapper automatically, or they can be created by the user
and passed into the mapper (or a mix of both). The advantage of that flexibility is that the mapper can call
replicas on totally different infrastructure (e.g. if you have two different clusters).

When the mapper creates the replicas, it creates duplicate envs
of the original mapped module's env (e.g. `env_replica_1`), and sends the module into the replica env on the same
cluster, thus creating many modules each in separate processes (and potentially separate nodes). Keep in mind that
you must specify the compute resources (e.g. `compute={"CPU": 0.5}`) in the `env` constructor if you have a multinode
cluster and want the replica envs to overflow onto worker nodes.

The mapper then simply calls each in a threadpool and collects the results. By default, the threadpool is the same
size as the number of replicas, so each thread blocks until a replica is available. You can control this by setting
`concurrency`, which is the number of simultaneous calls that can be made to any replica (e.g. if concurrency is 2,
then 2 threads will be calling each replica at the same time).

The mapper can either sit locally or on a cluster, but you should generally put it on the cluster if you can.
If the mapper is local, you'll need to send the mapped module to the cluster before passing it to the mapper,
and the mapper will create each new replica on the cluster remotely, which will take longer. The communication
between the mapper and the replicas will also be faster and more dependable if they are on the same cluster.
Just note that if you're sending or returning a large amount of data, it may take time to transfer before or after
you see the loading bar when the mapper is actually processing. Generally you'll get the best performance if you
read and write to blob storage or the filesystem rather than sending the data around.


Mapper Factory Method
~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: runhouse.mapper


Mapper Class
~~~~~~~~~~~~

.. autoclass:: runhouse.Mapper
   :members:
   :exclude-members:

    .. automethod:: __init__
