**Runhouse CLI**

Initial features:

- command for spinning up a python shell in a remote environment

*Sample commands*: 

To open a bash shell on remote execution environment based on the rh_1_gpu hardware spec:

```python -m runhouse -h rh_1_gpu```

To run a python file on remote environment based on the same hardware:

``python -m runhouse -h rh_1_gpu -f /Users/josh.l/dev/runhouse/runhouse/josh.py``

*(Note: for now need the python -m since we have not deployed runhouse as a PyPI package)*
