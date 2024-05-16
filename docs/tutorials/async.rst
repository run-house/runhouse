Asynchronous Programming
========================

.. raw:: html

    <p><a href="https://colab.research.google.com/github/run-house/notebooks/blob/stable/docs/async.ipynb">
    <img height="20px" width="117px" src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a></p>

Background
----------

*Note that this tutorial assumes basic understanding of Runhouse
Functions & Modules; it is recommended that you check out* `our
functions and modules
tutorial <https://www.run.house/docs/tutorials/api-modules>`__ *before
diving into this one.*

As we’ve discussed before, once you take a Python function or module and
send it to a Runhouse cluster, the cluster holds your resource in
memory, and each time that function or module is called by a client, it
simply accesses it in memory and calls it. Under the hood, we have a
fully asynchronous server running (FastAPI), and a separate process for
each environment where your Runhouse resources live. These processes all
have their own async event loops, and if you run synchronous functions
on Runhouse, they are ran in a separate thread to allow for many
concurrent calls to the cluster. **Note that if you are unfamiliar with
asynchronous programming in Python, you should just continue using
standard, Python sync functions and leave the performance to us**.

Native Async Functions
----------------------

But, what if you’re writing code that leverages Python’s powerful
asynchronous functionality? Luckily, we provide rich async support in a
variety of ways. First off, any function that is labeled with Python’s
async keyword, when sent to a Runhouse cluster, will be *executed within
the environment processes’s async event loop*, and not in a separate
thread. **This means that you should be very careful that you are not
running any costly, synchronous code within an async function, to avoid
blocking up your the event loop within your environment on the server.
Poorly written async functions will not block the entire Runhouse
daemon, but will block other functions within the same environment as
the user code.**

Client side, you also need to ``await`` a call to this function the same
way you would if the function was running locally. Let’s check out an
example. First, we’ll start a local Runhouse daemon to mess with:

.. code:: ipython3

    ! runhouse restart

Then, we’ll define a simple ``async`` function to send to Runhouse:

.. code:: ipython3

    async def async_test(time_to_sleep: int):
        import asyncio

        await asyncio.sleep(time_to_sleep)
        return time_to_sleep

We can send this to Runhouse the same way we would any other Runhouse
function:

.. code:: ipython3

    import runhouse as rh

    async_test_fn_remote = rh.function(async_test).to(rh.here)


.. parsed-literal::
    :class: code-output

    INFO | 2024-04-30 18:50:35.023995 | Because this function is defined in a notebook, writing it out to a file to make it importable. Please make sure the function does not rely on any local variables, including imports (which should be moved inside the function body). Functions defined in Python files can be used normally.
    INFO | 2024-04-30 18:50:35.060478 | Sending module async_test of type <class 'runhouse.resources.functions.function.Function'> to local Runhouse daemon


Then, we can call this function as we would if it were a local async
function. The network call to the remote cluster will execute
asynchronously within our local event loop (our code backed by
``httpx.AsyncClient``) and the async function itself will execute within
the async event loop on the remote server.

.. code:: ipython3

    await async_test_fn_remote(2)




.. parsed-literal::
    :class: code-output

    2



Voila! Async functions are supported the way you’d expect them to be.
There are a few other advanced cases, too:

Advanced: Running Sync Functions as Async Locally
-------------------------------------------------

There’s another important case that we support. Let’s say that your
standard, synchronous functions are running on a remote Runhouse
machine. When you call them from your local machine, there is inevitably
network I/O involved in communicating with the cluster. You may want to
not have your code block on this network call (for example if the
function takes a long time to execute), so that you can avoid blocking
your local Python code. You can choose to run this function
asynchronously, locally, and this allows you to get back a coroutine
from Runhouse, that you can then use to check if Note that this means
your local code will have to use async primitives, even though it is
calling what you defined as a sync function. Let’s check out an example
of this:

.. code:: ipython3

    def synchronous_sleep(time_to_sleep: int):
        import time

        time.sleep(time_to_sleep)
        return time_to_sleep

    sync_sleep_fn_remote = rh.function(synchronous_sleep).to(rh.here)


.. parsed-literal::
    :class: code-output

    INFO | 2024-04-30 18:57:00.533012 | Because this function is defined in a notebook, writing it out to a file to make it importable. Please make sure the function does not rely on any local variables, including imports (which should be moved inside the function body). Functions defined in Python files can be used normally.
    INFO | 2024-04-30 18:57:00.577673 | Sending module synchronous_sleep of type <class 'runhouse.resources.functions.function.Function'> to local Runhouse daemon


We can now call this function with the ``run_async`` argument set to to
``True``. This makes it not actually run locally immediately, and
instead returns a coroutine that you’d await, as if this function were
asynchronous. Note that, in your environment on your Runhouse cluster,
the functions runs in a thread, but the call to it locally is
asynchronous, and uses ``httpx.AsyncClient``.

.. code:: ipython3

    await sync_sleep_fn_remote(2, run_async=True)




.. parsed-literal::
    :class: code-output

    2



You could also use ``asyncio.create_task()`` to not block your code on
the execution and then ``await`` it when you want the result. When using
a function defined as async or a sync function with ``run_async=True``,
you always get back a coroutine, which you can do with what you please.

If I wanted, I could still call this function as a fully synchronous
function:

.. code:: ipython3

    sync_sleep_fn_remote(2)




.. parsed-literal::
    :class: code-output

    2



Advanced: Running Async Functions as Sync Locally
-------------------------------------------------

The third critical case that we support is mostly applicable when you’re
writing async code for the purpose of running it on the Runhouse
cluster, but want to make synchronous calls to the server. The reason
for you writing async code to run on the server is because our Runhouse
server uses ASGI and runs everything asynchronously, so you can take
advantage of the performance gains that come along with async code, but
call it locally as you would a normal client calling a normal server,
unaware of the backend implementation of the server. We can take the
same async function I defined earlier and call it synchronously:

.. code:: ipython3

    async_test_fn_remote(2, run_async=False)




.. parsed-literal::
    :class: code-output

    2



That’s all there is to it! We’ve tried our hardest to make working with
async code seamless from a user’s perspective. There are other edge
cases we’ve put time into supporting and we’re happy to discuss
architecture anytime – feel free to `file an issue on
Github <https://github.com/run-house/runhouse/issues>`__ or `join us on
Discord <https://discord.com/invite/RnhB6589Hs>`__ to discuss more!
