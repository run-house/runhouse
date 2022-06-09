**Runhouse CLI**

Initial features:

- command for spinning up a python shell in a remote environment

*Sample commands*: 

To open a bash shell on remote execution environment based on the rh_1_gpu (single GPU) hardware spec:

```python -m runhouse -h rh_1_gpu```

To run a python file on remote environment based on the same hardware:

``python -m runhouse -h rh_1_gpu -f ./training_script.py``

*(Note: for now running as python module since it has not yet been deployed as a package)*
