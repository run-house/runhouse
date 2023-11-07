# Runhouse Testing

Unit testing inside code files. How?
* Add methods called __test_<method>
* Configure pytest paths to spot those tests
* Running the codefile itself triggers unit tests
  * Test level is settable in codefile, but "unit" by default
    * "unit" - Mock and dryrun fixtures or unit testing only
    * "local" - Local fixtures only, generally in containers when possible
    * "minimal" - Minimal set of remote fixtures, used for iteration on a file (DEFAULT)
    * "thorough" - Thorough set of remote fixtures for PR approval
    * "maximal" - Testing with all possible fixtures
* Integration tests
  *

class:
    - test_on_this_cluster
    - test_local_function_to

Basic idea -
1) Allow a test to inherit cases from parent
   2) But still retain ability to run single test
2) Run tests based on which fixtures are being created (globally)
   3) Allow different test files to define different fixture levels
   4) Don't run duplicate tests (same fixtures)
4) Allow functions to easily be called on remote hardware
5) Allow new edge cases to be added easily
6) Allow more than one test suite to use a test (e.g. cluster and function sharing a "run function on cluster" test)
6) Define a way to manipulate one fixture with existing infra into another, with cases being sequential?

rh.Test:
  properties: mock_fixtures, local_fixtures, minimal, thorough, maximal
  mock_tests, local_tests, minimal_tests, etc.
  Fixture properties define global fixtures! e.g. if other fixtures in conftest must use the fixture values
  Don't use parameterize, or tests will run on each infra type sequentially (better to run all tests for a fixture)
    Maybe we can just reorder?
  @rh.tests.remote() decorator allows specifying where function is called (inc fixture/param name), but test module
    is only sent once.
  in main or pytest.ini, allow setting the fixtures or level, and only tests within the suite which take those fixtures
    (or derivative fixtures) run.

  Alternative: decorators for mock, local, etc. tests, and maybe fixtures
    Maybe allow skipping certain parent tests? e.g. with decorator or property


Example of implementation process for AWSLambdaFn:
* Define local and minimal fixtures in test_aws_lambda_fn.py
  * Implement test_mocks, test_local, etc., maybe starting with just super().test_mocks()
    * Inherit test cases from parents - Fn, Module, Resource
    * Ideally can also call tests just defined as test_* functions
* Start adding unit tests and functionality to aws_lambda_fn.py
  * Optional - review fixtures and tests with buddy
* Add feature and edge case tests to test_aws_lambda_fn.py
  * For each, add to relevant fixture lists (e.g. test_mocks, test_local, test_minimal, etc.)
* If changes to function.py or module.py are required, run test_function or test_module with "minimal"
  * Before merging, any materially modified files had tests run with "thorough"
    * If any breakage is found for a fixture (e.g. sagemaker), jump into culprit's test file and run with "minimal"
  * CI/CD should run full suite with local before merging
  * After running tests, results overwrite cells in a file called test_record.txt (which can be compared with last commit to see changes)
* Commit

Example of implementation process for HTTPS:
* Add a new file test_https_cluster with TestHTTPSCluster, subclass of TestCluster
  * Set minimal to cpu cluster launched with open port (or manipulate existing one?). Set thorough to include static with HTTPS.
  * Create a test case to call the https function inside a docker container, i.e. a fresh environment
* Iterate with minimal, then try thorough.
* Add a local docker HTTPS cluster fixture to enable local level (or maybe iterate with local and unit first)
* Set TestCluster to include https_cluster under thorough (maybe make sure others like TestFunction ingest it from that)
