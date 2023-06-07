import unittest
from pprint import pprint

import pytest

import runhouse as rh

FUNC_RUN_NAME = "my_test_run"
ASYNC_RUN_NAME = "my_async_run"
CTX_MGR_RUN = "my_run_activity"

CLI_RUN_NAME = "my_cli_run"
S3_BUCKET = "runhouse-runs"

PATH_TO_CTX_MGR_RUN = f"{rh.Run.LOCAL_RUN_PATH}/{CTX_MGR_RUN}"


def setup():
    from runhouse.rns.api_utils.utils import create_s3_bucket

    create_s3_bucket(S3_BUCKET)


# ------------------------- FUNCTION RUN ----------------------------------


@pytest.mark.clustertest
def test_create_run_on_cluster(summer_func):
    """Initializes a Run, which will run synchronously on the cluster.
    Returns the function's result"""
    res = summer_func(1, 2, name_run=FUNC_RUN_NAME)
    assert res == 3


@pytest.mark.clustertest
def test_read_fn_stdout(cpu_cluster):
    """Reads the stdout for the Run."""
    fn_run = cpu_cluster.get_run(FUNC_RUN_NAME)
    stdout = fn_run.stdout()
    pprint(stdout)
    assert stdout


@pytest.mark.clustertest
def test_load_run_result(cpu_cluster):
    """Load the Run created above directly from the cluster."""
    # Note: Run only exists on the cluster (hasn't yet been saved to RNS).
    func_run = cpu_cluster.get_run(FUNC_RUN_NAME)
    assert func_run.result() == 3


@pytest.mark.clustertest
def test_get_or_call_from_cache(summer_func):
    """Cached version of synchronous run - if already completed return the result, otherwise run and wait for
    completion before returning the result."""
    # Note: In this test since we already ran the function, it should immediately return the result
    run_output = summer_func.get_or_call(name_run=FUNC_RUN_NAME)

    assert run_output == 3


@pytest.mark.clustertest
def test_get_or_call_no_cache(summer_func):
    """Cached version of synchronous run - if already completed return the result, otherwise run and wait for
    completion before returning the result."""
    # Note: In this test since we do not have a run with this name, it should first execute the function
    run_output = summer_func.get_or_call(name_run="another_sync_run", a=1, b=2)

    assert run_output == 3


@unittest.skip("Not implemented yet.")
@pytest.mark.clustertest
def test_get_or_call_latest(summer_func):
    """Cached version of synchronous run - if already completed return the result, otherwise run and wait for
    completion before returning the result."""
    # Note: In this test since we are providing a name of "latest", it should return the latest cached version
    run_output = summer_func.get_or_call("latest")

    assert run_output == 3


@pytest.mark.clustertest
def test_run_fn_async(summer_func):
    # TODO [JL] Failed - "Error inside function remote: Only one live display may be active at once"
    """Execute function async on the cluster. If a run already exists, do not re-run. Returns a Run object."""
    async_run = summer_func.run(name_run=ASYNC_RUN_NAME, a=1, b=2)

    assert isinstance(async_run, rh.Run)


@unittest.skip("Not implemented yet.")
@pytest.mark.clustertest
def test_send_run_to_system_on_completion(summer_func):
    # Only once the run actually finishes do we send to S3
    async_run = summer_func.run(name_run=ASYNC_RUN_NAME, a=1, b=2).to(
        "s3", on_completion=True
    )

    assert isinstance(async_run, rh.Run)


@pytest.mark.clustertest
def test_get_or_run(summer_func):
    """Execute function async on the cluster. If a run already exists, do not re-run. Returns a Run object."""
    # Note: In this test since we already ran the function with this run name, will immediately return the Run object.
    async_run = summer_func.get_or_run(name_run="async_get_or_run", a=1, b=2)
    assert isinstance(async_run, rh.Run)


@pytest.mark.clustertest
def test_run_refresh(slow_func):
    from runhouse.rns.run import RunStatus

    async_run = slow_func.get_or_run(name_run="async_get_or_run")

    while async_run.refresh().status in [RunStatus.RUNNING, RunStatus.NOT_STARTED]:
        # do stuff .....
        pass

    assert async_run.refresh().status == RunStatus.COMPLETED


@pytest.mark.clustertest
def test_get_async_run_result(summer_func):
    """Read the results from an async run."""
    async_run = summer_func.get_or_run(name_run=ASYNC_RUN_NAME)
    assert isinstance(async_run, rh.Run)
    assert async_run.result() == 3


@pytest.mark.clustertest
def test_get_or_run_no_cache(summer_func):
    """Execute function async on the cluster. If a run already exists, do not re-run. Returns a Run object.
    Note: In this test since no Run exists with this name, will trigger the function async on the cluster and in the
    meantime return a Run object."""
    async_run = summer_func.get_or_run(name_run="new_async_run", a=1, b=2)
    assert isinstance(async_run, rh.Run)


@unittest.skip("Not implemented yet.")
@pytest.mark.clustertest
def test_get_or_run_latest(summer_func):
    """Execute function async on the cluster. If a run already exists, do not re-run. Returns a Run object."""
    # Note: In this test since we are providing "latest", will return the latest cached version.
    async_run = summer_func.get_or_run(name_run="latest")
    assert isinstance(async_run, rh.Run)


@pytest.mark.clustertest
def test_delete_async_run_from_system(cpu_cluster):
    async_run = cpu_cluster.get_run(ASYNC_RUN_NAME)
    async_run.folder.rm()
    assert not async_run.folder.exists_in_system()


@pytest.mark.clustertest
@pytest.mark.rnstest
def test_save_fn_run_to_rns(cpu_cluster):
    """Saves run config to RNS"""
    func_run = cpu_cluster.get_run(FUNC_RUN_NAME)
    assert func_run

    func_run.save(name=FUNC_RUN_NAME)
    loaded_run = rh.Run.from_name(name=FUNC_RUN_NAME)
    assert rh.exists(loaded_run.name, resource_type=rh.Run.RESOURCE_TYPE)


@pytest.mark.clustertest
def test_create_anon_run_on_cluster(summer_func):
    """Create a new Run without giving it an explicit name."""
    # Note: this will run synchronously and return the result
    res = summer_func(1, 2, name_run=True)
    assert res == 3


@unittest.skip("Not yet implemented.")
@pytest.mark.clustertest
def test_latest_fn_run(summer_func):
    run_output = summer_func.get_or_call(run_str="latest")

    assert run_output == 3


@pytest.mark.clustertest
def test_copy_fn_run_from_cluster_to_local(cpu_cluster):
    my_run = cpu_cluster.get_run(FUNC_RUN_NAME)
    my_local_run = my_run.to("here")
    assert my_local_run.folder.exists_in_system()


@pytest.mark.clustertest
@pytest.mark.awstest
def test_copy_fn_run_from_system_to_s3(cpu_cluster):
    my_run = cpu_cluster.get_run(FUNC_RUN_NAME)
    my_run_on_s3 = my_run.to("s3", path=f"/{S3_BUCKET}/my_test_run")

    assert my_run_on_s3.folder.exists_in_system()

    # Delete the run from s3
    my_run_on_s3.folder.rm()
    assert not my_run_on_s3.folder.exists_in_system()


@pytest.mark.clustertest
def test_read_fn_run_inputs_and_outputs():
    my_run = rh.Run.from_name(name=FUNC_RUN_NAME)
    inputs = my_run.inputs()
    assert inputs == {"args": [1, 2], "kwargs": {}}

    output = my_run.result()
    assert output == 3


@pytest.mark.rnstest
def test_read_fn_stdout_from_rns_run():
    # Read the stdout saved when running the function on the cluster
    my_run = rh.Run.from_name(name=FUNC_RUN_NAME)
    stdout = my_run.stdout()
    pprint(stdout)
    assert stdout


@pytest.mark.rnstest
def test_delete_fn_run_from_rns():
    func_run = rh.Run.from_name(FUNC_RUN_NAME)
    func_run.delete_configs()
    assert not rh.exists(name=func_run.name, resource_type=rh.Run.RESOURCE_TYPE)


@pytest.mark.clustertest
def test_slow_running_async_fn_run(cpu_cluster, slow_func):
    """Run a function that takes a long time to run - since we are running this async we should
    immediately get back the run object with a status of RUNNING."""
    from runhouse.rns.run import RunStatus

    async_run = slow_func.run(name_run="my_slow_async_run", a=1, b=2)

    assert isinstance(async_run, rh.Run)
    assert async_run.refresh().status == RunStatus.RUNNING


@pytest.mark.clustertest
def test_slow_running_fn_run(cpu_cluster, slow_func):
    """Run a function that takes a long time to run - this waits for the execution to complete on the cluster
    before returning the result."""
    slow_run_name = "my_slow_run"
    run_res = slow_func(2, 2, name_run=slow_run_name)
    assert run_res == 4

    func_run = cpu_cluster.get_run(slow_run_name)
    assert func_run

    func_run.folder.rm()
    assert not func_run.folder.exists_in_system()


# ------------------------- CLI RUN ------------ ----------------------


@pytest.mark.clustertest
def test_create_cli_python_command_run(cpu_cluster):
    # Run python commands on the specified system. Save the run results to the .rh/logs/<run_name> folder of the system.
    return_codes = cpu_cluster.run_python(
        [
            "import runhouse as rh",
            "import logging",
            "cpu = rh.cluster('^rh-cpu')",
            "logging.info(f'On the cluster {cpu.name}!')",
            "rh.cluster('^rh-cpu').save()",
        ],
        name_run=CLI_RUN_NAME,
    )

    assert "INFO" in return_codes


@pytest.mark.clustertest
def test_create_cli_command_run(cpu_cluster):
    """Run CLI command on the specified system.
    Saves the run results to the .rh/logs/<run_name> folder of the system."""
    cmd_stdout = cpu_cluster.run(["python --version"], name_run=CLI_RUN_NAME)
    assert cmd_stdout.strip() == "Python 3.10.6"


@pytest.mark.clustertest
def test_load_cli_command_run_from_cluster(cpu_cluster):
    # Run only exists on the cluster (hasn't yet been saved to RNS).
    cli_run = cpu_cluster.get_run(CLI_RUN_NAME)
    assert isinstance(cli_run, rh.Run)


@pytest.mark.clustertest
@pytest.mark.rnstest
def test_save_cli_run_on_cluster_to_rns(cpu_cluster):
    cli_run = cpu_cluster.get_run(CLI_RUN_NAME)
    cli_run.save(name=CLI_RUN_NAME)

    loaded_run_from_rns = rh.Run.from_name(name=CLI_RUN_NAME)
    assert loaded_run_from_rns


@pytest.mark.clustertest
def test_read_cli_command_stdout():
    # Read the stdout from the system the command was run
    cli_run = rh.Run.from_name(name=CLI_RUN_NAME)
    cli_stdout = cli_run.stdout()
    assert cli_stdout == "Python 3.10.6"


@pytest.mark.clustertest
def test_delete_cli_run_from_system(cpu_cluster):
    cli_run = cpu_cluster.get_run(CLI_RUN_NAME)
    cli_run.folder.rm()

    assert not cli_run.folder.exists_in_system()


@pytest.mark.rnstest
def test_cli_run_exists_in_rns():
    cli_run = rh.Run.from_name(name=CLI_RUN_NAME)
    assert rh.exists(cli_run.name, resource_type=rh.Run.RESOURCE_TYPE)


@pytest.mark.rnstest
def test_delete_cli_run_from_rns():
    cli_run = rh.Run.from_name(CLI_RUN_NAME)
    cli_run.delete_configs()
    assert not rh.exists(name=cli_run.name, resource_type=rh.Run.RESOURCE_TYPE)


# ------------------------- CTX MANAGER RUN ----------------------------------


@pytest.mark.clustertest
@pytest.mark.rnstest
def test_create_local_ctx_manager_run(summer_func):
    from runhouse.rh_config import rns_client

    with rh.run(name=CTX_MGR_RUN) as r:
        # Add all Runhouse objects loaded or saved in the context manager to the Run's artifact registry
        # (upstream + downstream artifacts)
        my_func = rh.Function.from_name("summer_func")
        my_func.save("my_new_func")

        my_func(1, 2, name_run="my_new_run")

        current_run = my_func.system.get_run("my_new_run")
        run_res = current_run.result()
        print(f"Run result: {run_res}")

        cluster = rh.load(name="@/rh-cpu")
        print(f"Cluster loaded: {cluster.name}")

    print(f"Saved Run with name: {r.name} to path: {r.folder.path}")

    # Artifacts include the rns resolved name (ex: "/jlewitt1/rh-cpu")
    expected_downstream = [
        rns_client.resolve_rns_path(a) for a in r.downstream_artifacts
    ]
    expected_upstream = [rns_client.resolve_rns_path(a) for a in r.upstream_artifacts]

    assert r.downstream_artifacts == expected_downstream
    assert r.upstream_artifacts == expected_upstream


@pytest.mark.localtest
def test_load_named_ctx_manager_run():
    ctx_run = rh.Run.from_path(path=PATH_TO_CTX_MGR_RUN)
    assert ctx_run.folder.exists_in_system()


@pytest.mark.localtest
def test_read_stdout_from_ctx_manager_run():
    ctx_run = rh.Run.from_path(path=PATH_TO_CTX_MGR_RUN)
    stdout = ctx_run.stdout()
    pprint(stdout)
    assert stdout


@pytest.mark.rnstest
def test_save_ctx_run_to_rns():
    ctx_run = rh.Run.from_path(path=PATH_TO_CTX_MGR_RUN)
    ctx_run.save()
    assert rh.exists(name=ctx_run.name, resource_type=rh.Run.RESOURCE_TYPE)


@pytest.mark.clustertest
def test_delete_run_from_system():
    ctx_run = rh.Run.from_path(path=PATH_TO_CTX_MGR_RUN)
    ctx_run.folder.rm()
    assert not ctx_run.folder.exists_in_system()


@pytest.mark.rnstest
def test_delete_run_from_rns():
    ctx_run = rh.Run.from_name(CTX_MGR_RUN)
    ctx_run.delete_configs()
    assert not rh.exists(name=ctx_run.name, resource_type=rh.Run.RESOURCE_TYPE)


if __name__ == "__main__":
    setup()
    unittest.main()
