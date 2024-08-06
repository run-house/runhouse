FROM python:3.9

WORKDIR /code

# Copy only the file with the requirements first, not the rest of the code.
# As this file doesn't change often, Docker will detect it and use the cache for this step,
# enabling the cache for the next step too.
COPY ./requirements.txt /code/requirements.txt

# The --no-cache-dir option tells pip to not save the downloaded packages locally,
# as that is only if pip was going to be run again to install the same packages,
# but that's not the case when working with containers.
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# As this has all the code which is what changes most frequently the Docker
# cache won't be used for this or any following steps easily.
# So, it's important to put this near the end of the Dockerfile,
# to optimize the container image build times.
COPY ./app /code/app

# Set the command to use fastapi run, which uses Uvicorn underneath.
# CMD takes a list of strings, each of these strings is
# what you would type in the command line separated by spaces.
# This command will be run from the current working directory,
# the same /code directory you set above with WORKDIR /code.
CMD ["fastapi", "run", "app/main.py", "--port", "80"]
