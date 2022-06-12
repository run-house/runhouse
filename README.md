**Runhouse CLI**

Initial features:

- command for spinning up a bash shell in a remote environment
- command for registering a URI for python code
- command for running python code in a remote environment on specified hardware 


*To download the runhouse package*

``pip install runhouse-0.1.0-py3-none-any.whl``

*Sample commands*: 

To open a bash shell on remote execution environment based on the default rh_1_gpu (single GPU) hardware spec:

```runhouse shell```

To run a python file on remote environment based on the same hardware:

``runhouse -p runhouse/scripts/training_script.py run``

To register a URI for the code in a specific file based on specified hardware:

``runhouse -h rh_1_gpu runhouse -p runhouse/scripts/training_script.py register --user josh --name bert-preprocessing``

*Support*: 

``runhouse --help``

``runhouse --version``