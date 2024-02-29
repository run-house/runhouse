# Runhouse Testing

Runhouse's testing is demanding and complex. We need to test across an ultra-wide range of infra, but can't simply
test each infra locale separately because most of our features and changes are cross-cutting. Without a surgical
approach, this becomes a cartesian explosion quickly. We use testing "levels" to define which fixtures to use
for a given test suite, and which tests to run for a set of fixtures:
  * "unit" - Mock and dryrun fixtures (often a single local mock) and unit testing only
  * "local" - A comprehensive set of local fixtures, often in containers (run in CI/CD)
  * "minimal" - Minimal set of remote and local fixtures, used for iteration and most PR merges (DEFAULT)
  * "release" - Thorough set of fixtures, run ahead of release (or PR merge for major changes)
  * "maximal" - Testing with all possible fixtures, rarely used

This allows us to run a single test file with different levels to test different infra, and override the level
fixtures for a given test suite if we want to test it more precisely or thoroughly. For example, the default minimal
cluster fixture for "test_cluster" might just be a static cpu cluster, but for "test_sagemaker" would be a cpu
SageMaker cluster. We also make use of test classes and inheritance to allow us to
easily add new infra to the test suite and inherit a large portion of the test cases, while still being able to
do TDD and add edge cases one-off as needed. We also use imports within test modules or classes to reuse
test cases in multiple suites where appropriate (just importing the test case function is enough for PyTest to
do this, so ignore warnings that the imports are unused).

Runhouse uses Sky (which uses ssh) to communicate with the clusters it runs on (e.g. AWS EC2). Sky generates
a keypair for you locally the first time you use it to communicate with a cluster, and for local container tests
we take this key from `~/.ssh/sky-key` and copy it to the Docker containers such that Sky can communicate with them. You can override the choice of keypair by adding the following line to your `~/.rh/config.yaml`: `default_keypair: <path to your private key>`.

To run a single test file with a given level, use a command like this:
```bash
pytest -s -v --level "unit" tests/test_resources/test_resource.py
```

Or a specific test:
```bash
pytest -s -v --level "unit" tests/test_resources/test_resource.py::TestResource::test_save_and_load
```

To run tests across multiple suites but only including a specific fixture, you can do:
```bash
pytest -s -v --level "local" -k "docker_cluster_pk_ssh_no_auth" tests
```
Make sure the fixture(s) of interest are included in the level you're running at, or they won't natch with any tests.
You can also exclude fixtures with "-m" and a tilde, e.g. "-m ~docker_cluster_public_key".

## Sample Workflow

Say you're adding a new type of infra behind an existing Resource abstraction in Runhouse, perhaps an AWSLambdaFn
which is a subclass of Function. Here's an example of what your workflow might look like:

* Create a new directory test_aws_lambda underneath test_functions
* Define some initial fixtures (likely minimal or local) in test_aws_lambda/conftest.py
* Create a TestAWSLambdaFn class in test_aws_lambda/test_aws_lambda_fn.py
  * Begin working through the inherited test cases from parents - Fn, Module, Resource - trying to get to passing with your new fixtures
* Start adding AWS Lambda-specific unit, feature, and edge case tests to test_aws_lambda_fn.py
  * Optional - review fixtures and tests with reviewer
  * Start unblocking the tests on each fixture level
* If changes to function.py or module.py are required, run test_function or test_module with "minimal"
  * Before merging, run with level "release" if any cross-cutting modules are materially modified
    * If any breakage is found for a fixture (e.g. sagemaker), jump into culprit's test file and run with "minimal"
* Iterate with reviewer and rerun tests at various levels as needed
* CI/CD will run the full Runhouse suite with local before merging
* Merge
