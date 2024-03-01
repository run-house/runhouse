Deploy Stable Diffusion XL 1.0 on AWS EC2
================================================

This simple code example brings up an AWS EC2 instance using SkyPilot.

Install dependencies
--------------------
.. code-block:: cli

    # Optionally, set up a virtual environment
    conda create -n rh-sdxl python=3.9.15
    conda activate rh-sdxl

    # Install the required packages
    pip install -r requirements.txt


Setup credentials
-----------------

We need both AWS secrets and HuggingFace secrets to run this example.

.. code-block:: cli

    # Set up your AWS credentials file
    aws configure

    # Skypilot should tell you whether it can launch instances on AWS
    sky check

    # Set up your HuggingFace credentials
    export HF_TOKEN=<your token>

Run the example
---------------
.. code-block:: cli

    python sdxl.py
