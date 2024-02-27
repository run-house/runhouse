Stable Diffusion XL Pipeline on AWS Inferentia2
================================================

This simple code example brings up an AWS Inferentia2 instance using SkyPilot.

Install dependencies
--------------------
.. code-block:: cli

    # Optionally, set up a virtual environment
    conda create -n rh-inf2 python=3.9.15
    conda activate rh-inf2

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

    python inf2_sdxl.py
