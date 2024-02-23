# Contributing to Runhouse
Please file an [issue](https://github.com/run-house/runhouse/issues) if you encounter a bug.

If you would like to submit a bug-fix or improve an existing feature, please submit a pull request following the
process outlined below.

If you would like to contribute, but don't know what to add, you can look for open issues labeled
`good first issue`, or take a look at the [funhouse repo](https://github.com/run-house/funhouse) to
create and add your own ML application using Runhouse!

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

To run tests, please install test/requirements.txt.
```
pip install -r tests/requirements.txt
```

Additional optional packages to install to run related tests:

aws related tests
```
pip install -r tests/test_requirements/aws_test_requirements.txt
```

google related tests
```
pip install -r tests/test_requirements/google_tests_requirements.txt
```



## Documentation
Docs source code is located in `docs/`. To build and review docs locally:

```
pip install -r docs/requirements.txt
cd docs/
make clean html
```

### Tutorials and Examples
Notebook (`.ipynb`) code lives in [run-house/notebooks](https://github.com/run-house/notebooks). If modifying
a tutorial or example involving a `.ipynb` file, please refer to these
[instructions](https://github.com/run-house/notebooks?tab=readme-ov-file#syncing-docs-to-run-houserunhouse) for
how to upload your notebook to the notebooks repo and sync the rendered `.rst` file over to the runhouse repo.
