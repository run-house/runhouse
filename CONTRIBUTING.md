# Contributing to Runhouse
Please file an [issue](https://github.com/run-house/runhouse/issues) if you encounter a bug.

If you would like to submit a bug-fix or improve an existing feature, please submit a pull request following the
process outlined below.

## Development Process
If you want to modify code, please follow the instructions for creating a Pull Request.

1. Fork the Github repository, and then clone the forked repo to local.
```
git clone git@github.com:<your username>/runhouse.git
cd runhouse
git remote add upstream https://github.com/run-house/runhouse.git
```

2. Create a new branch for your development changes:
```
git checkout -b branch-name
```

3. Install Runhouse
```
pip install -e .
```

4. Develop your features

5. Download and run pre-commit to automatically format your code using black and ruff.

```
pip install pre-commit
pre-commit run --files [FILES [FILES ...]]
```

6. Add, commit, and push your changes. Create a "Pull Request" on GitHub to submit the changes for review.

```
git push -u origin branch-name
```

## Testing

To run tests, please install `pytest`.
```
pip install pytest
```

Additional optional packages to install to run related tests:

aws related tests
* `s3fs==2023.1.0`
* `awscli==1.25.60`
* `boto3==1.24.59`
* `pycryptodome==3.12.0`

google related tests
* `google-api-python-client`
* `google-cloud-storage`
* `gcsfs`

table tests
* `datasets`
* `dask`

## Documentation
Docs source code is located in `docs/`. To build and review docs locally:

```
pip install -r docs/requirements.txt
cd docs/
make clean html
```
