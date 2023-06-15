import unittest
from pathlib import Path
from pprint import pprint

import pytest
import runhouse as rh

CTX_MGR_RUN = "my_run_activity"
CLI_RUN_NAME = "my_cli_run"

PATH_TO_CTX_MGR_RUN = f"{rh.Run.LOCAL_RUN_PATH}/{CTX_MGR_RUN}"

RUN_FILES = (
    rh.Run.INPUTS_FILE,
    rh.Run.RESULT_FILE,
    rh.Run.RUN_CONFIG_FILE,
    ".out",
    ".err",
)


# ------------------------- FUNCTION RUN ----------------------------------


@pytest.mark.clustertest
@pytest.mark.runstest
def test_read_fn_stdout(cpu_cluster, submitted_run):
    """Reads the stdout for the Run."""
    fn_run = cpu_cluster.get_run(submitted_run)
    stdout = fn_run.stdout()
    pprint(stdout)
    assert stdout


@pytest.mark.clustertest
@pytest.mark.runstest
def test_load_run_result(cpu_cluster, submitted_run):
    """Load the Run created above directly from the cluster."""
    # Note: Run only exists on the cluster (hasn't yet been saved to RNS).
    func_run = cpu_cluster.get_run(submitted_run)
    assert func_run.result() == 3


@pytest.mark.clustertest
@pytest.mark.runstest
def test_get_or_call_from_cache(summer_func, submitted_run):
    """Cached version of synchronous run - if already completed return the result, otherwise run and wait for
    completion before returning the result."""
    # Note: In this test since we already ran the function, it should return the result without re-running
    run_output = summer_func.get_or_call(run_name=submitted_run)
    assert run_output == 3


@pytest.mark.clustertest
@pytest.mark.runstest
def test_get_or_call_no_cache(summer_func):
    """Cached version of synchronous run - if already completed return the result, otherwise run and wait for
    completion before returning the result."""
    # Note: In this test since we do not have a run with this name, it should first execute the function
    # before returning its result
    run_output = summer_func.get_or_call(run_name="another_sync_run", a=1, b=2)
    assert run_output == 3


@pytest.mark.clustertest
@pytest.mark.runstest
def test_invalid_fn_sync_run(summer_func, cpu_cluster):
    """Test error handling for invalid function Run. The function expects to receive integers but
    does not receive any. An error should be thrown via Ray."""
    import ray

    try:
        summer_func.get_or_call(run_name="invalid_run")
    except ray.exceptions.RayTaskError as e:
        assert (
            str(e.args[0])
            == "summer() missing 2 required positional arguments: 'a' and 'b'"
        )


@pytest.mark.clustertest
@pytest.mark.runstest
def test_invalid_fn_async_run(summer_func):
    """Test error handling for invalid function Run. The function expects to receive integers but
    does not receive any. The Run object returned should have a status of `ERROR`, and the
    result should be its stderr."""
    run_obj = summer_func.get_or_run(run_name="invalid_async_run")

    assert run_obj.refresh().status == rh.RunStatus.ERROR
    assert "summer() missing 2 required positional arguments" in run_obj.result()


@unittest.skip("Not implemented yet.")
@pytest.mark.clustertest
@pytest.mark.runstest
def test_get_or_call_latest(summer_func):
    """Cached version of synchronous run - if already completed return the result, otherwise run and wait for
    completion before returning the result."""
    # Note: In this test since we are providing a name of "latest", it should return the latest cached version
    run_output = summer_func.get_or_call("latest")

    assert run_output == 3


@unittest.skip("Not implemented yet.")
@pytest.mark.clustertest
@pytest.mark.runstest
def test_send_run_to_system_on_completion(summer_func, submitted_async_run):
    # Only once the run actually finishes do we send to S3
    async_run = summer_func.run(run_name=submitted_async_run, a=1, b=2).to(
        "s3", on_completion=True
    )

    assert isinstance(async_run, rh.Run)


@pytest.mark.clustertest
@pytest.mark.runstest
def test_run_refresh(slow_func):
    async_run = slow_func.get_or_run(run_name="async_get_or_run", a=1, b=2)

    while async_run.refresh().status in [
        rh.RunStatus.RUNNING,
        rh.RunStatus.NOT_STARTED,
    ]:
        # do stuff .....
        pass

    assert async_run.refresh().status == rh.RunStatus.COMPLETED


@pytest.mark.clustertest
@pytest.mark.runstest
def test_get_async_run_result(summer_func, submitted_async_run):
    """Read the results from an async run."""
    async_run = summer_func.get_or_run(run_name=submitted_async_run)
    assert isinstance(async_run, rh.Run)
    assert async_run.result() == 3


@pytest.mark.clustertest
@pytest.mark.runstest
def test_get_or_run_no_cache(summer_func):
    """Execute function async on the cluster. If a run already exists, do not re-run. Returns a Run object."""
    # Note: In this test since no Run exists with this name, will trigger the function async on the cluster and in the
    # meantime return a Run object.
    async_run = summer_func.get_or_run(run_name="new_async_run", a=1, b=2)
    assert isinstance(async_run, rh.Run)

    run_result = async_run.result()
    assert run_result == 3


@unittest.skip("Not implemented yet.")
@pytest.mark.clustertest
@pytest.mark.runstest
def test_get_or_run_latest(summer_func):
    """Execute function async on the cluster. If a run already exists, do not re-run. Returns a Run object."""
    # Note: In this test since we are providing "latest", will return the latest cached version.
    async_run = summer_func.get_or_run(run_name="latest")
    assert isinstance(async_run, rh.Run)


@pytest.mark.clustertest
@pytest.mark.runstest
def test_delete_async_run_from_system(cpu_cluster, submitted_async_run):
    async_run = cpu_cluster.get_run(submitted_async_run)
    async_run.folder.rm()
    assert not async_run.folder.exists_in_system()


@pytest.mark.clustertest
@pytest.mark.rnstest
@pytest.mark.runstest
def test_save_fn_run_to_rns(cpu_cluster, submitted_run):
    """Saves run config to RNS"""
    func_run = cpu_cluster.get_run(submitted_run)
    assert func_run

    func_run.save(name=submitted_run)
    loaded_run = rh.run(name=submitted_run)
    assert rh.exists(loaded_run.name, resource_type=rh.Run.RESOURCE_TYPE)
    assert loaded_run.status == rh.RunStatus.COMPLETED


@pytest.mark.clustertest
@pytest.mark.runstest
def test_create_anon_run_on_cluster(summer_func):
    """Create a new Run without giving it an explicit name."""
    # Note: this will run synchronously and return the result
    res = summer_func(1, 2)
    assert res == 3


@unittest.skip("Not yet implemented.")
@pytest.mark.clustertest
@pytest.mark.runstest
def test_latest_fn_run(summer_func):
    run_output = summer_func.get_or_call(run_str="latest")
    assert run_output == 3


@pytest.mark.clustertest
@pytest.mark.runstest
def test_copy_fn_run_from_cluster_to_local(cpu_cluster, submitted_run):
    my_run = cpu_cluster.get_run(submitted_run)
    my_local_run = my_run.to("here")
    assert my_local_run.folder.exists_in_system()

    # Check that all files were copied
    folder_contents = my_local_run.folder.ls()
    for f in folder_contents:
        file_extension = Path(f).suffix
        file_name = f.split("/")[-1]
        assert file_extension in file_name or file_name in RUN_FILES


@pytest.mark.clustertest
@pytest.mark.awstest
@pytest.mark.runstest
def test_copy_fn_run_from_system_to_s3(cpu_cluster, runs_s3_bucket, submitted_run):
    my_run = cpu_cluster.get_run(submitted_run)
    my_run_on_s3 = my_run.to("s3", path=f"/{runs_s3_bucket}/my_test_run")

    assert my_run_on_s3.folder.exists_in_system()

    # Check that all files were copied
    folder_contents = my_run_on_s3.folder.ls()
    for f in folder_contents:
        file_extension = Path(f).suffix
        file_name = f.split("/")[-1]
        assert file_extension in file_name or file_name in RUN_FILES

    # Delete the run from s3
    my_run_on_s3.folder.rm()
    assert not my_run_on_s3.folder.exists_in_system()


@pytest.mark.clustertest
@pytest.mark.runstest
def test_read_fn_run_inputs_and_outputs(submitted_run):
    my_run = rh.run(name=submitted_run)
    inputs = my_run.inputs()
    assert inputs == {"args": [1, 2], "kwargs": {}}

    output = my_run.result()
    assert output == 3


@pytest.mark.rnstest
@pytest.mark.runstest
def test_delete_fn_run_from_rns(submitted_run):
    func_run = rh.run(submitted_run)
    func_run.delete_configs()
    assert not rh.exists(name=func_run.name, resource_type=rh.Run.RESOURCE_TYPE)


@pytest.mark.clustertest
@pytest.mark.runstest
def test_get_fn_status_updates(cpu_cluster, slow_func):
    """Run a function that takes a long time to run, confirming that its status changes as we refresh the Run"""
    async_run = slow_func.run(run_name="my_slow_async_run", a=1, b=2)

    assert isinstance(async_run, rh.Run)

    assert async_run.status == rh.RunStatus.RUNNING

    while async_run.refresh().status != rh.RunStatus.COMPLETED:
        # ... do something else while we wait for the run to finish
        pass

    assert async_run.refresh().status == rh.RunStatus.COMPLETED


# ------------------------- CLI RUN ------------ ----------------------


@pytest.mark.clustertest
@pytest.mark.runstest
def test_create_cli_python_command_run(cpu_cluster):
    # Run python commands on the specified system. Save the run results to the .rh/logs/<run_name> folder of the system.
    return_codes = cpu_cluster.run_python(
        [
            "import runhouse as rh",
            "import pickle",
            "import logging",
            "local_blob = rh.blob(name='local_blob', data=pickle.dumps(list(range(50))), mkdir=True).write()",
            "logging.info(f'Blob path: {local_blob.path}')",
            "local_blob.rm()",
        ],
        run_name=CLI_RUN_NAME,
    )
    pprint(return_codes)

    assert return_codes[0][0] == 0, "Failed to run python commands"
    assert "Blob path" in return_codes[0][1].strip()


@pytest.mark.clustertest
@pytest.mark.runstest
def test_create_cli_command_run(cpu_cluster):
    """Run CLI command on the specified system.
    Saves the Run locally to the rh/<run_name> folder of the local file system."""
    return_codes = cpu_cluster.run(["python --version"], run_name=CLI_RUN_NAME)

    assert return_codes[0][0] == 0, "Failed to run CLI command"
    assert return_codes[0][1].strip() == "Python 3.10.6"


@pytest.mark.clustertest
@pytest.mark.runstest
def test_send_cli_run_to_cluster(cpu_cluster):
    """Send the CLI based Run which was initially saved locally to the cpu cluster."""
    loaded_run = rh.run(name=CLI_RUN_NAME)
    assert loaded_run.status == rh.RunStatus.COMPLETED
    assert loaded_run.stdout() == "Python 3.10.6"

    # Save to default path on the cluster (~/.rh/logs/<run_name>)
    cluster_run = loaded_run.to(
        cpu_cluster, path=rh.Run._base_cluster_folder_path(name=CLI_RUN_NAME)
    )
    assert cluster_run.folder.exists_in_system()
    assert isinstance(cluster_run.folder.system, rh.Cluster)


@pytest.mark.clustertest
@pytest.mark.runstest
def test_load_cli_command_run_from_cluster(cpu_cluster):
    # Run only exists on the cluster (hasn't yet been saved to RNS).
    cli_run = cpu_cluster.get_run(CLI_RUN_NAME)
    assert isinstance(cli_run, rh.Run)


@pytest.mark.clustertest
@pytest.mark.rnstest
@pytest.mark.runstest
def test_save_cli_run_on_cluster_to_rns(cpu_cluster):
    cli_run = cpu_cluster.get_run(CLI_RUN_NAME)
    cli_run.save(name=CLI_RUN_NAME)

    loaded_run_from_rns = rh.run(name=CLI_RUN_NAME)
    assert loaded_run_from_rns


@pytest.mark.rnstest
@pytest.mark.runstest
def test_read_cli_command_stdout():
    # Read the stdout from the system the command was run
    cli_run = rh.run(name=CLI_RUN_NAME)
    cli_stdout = cli_run.stdout()
    assert cli_stdout == "Python 3.10.6"


@pytest.mark.clustertest
@pytest.mark.runstest
def test_delete_cli_run_from_system(cpu_cluster):
    cli_run = cpu_cluster.get_run(CLI_RUN_NAME)
    cli_run.folder.rm()

    assert not cli_run.folder.exists_in_system()


@pytest.mark.rnstest
@pytest.mark.runstest
def test_cli_run_exists_in_rns():
    cli_run = rh.run(name=CLI_RUN_NAME)
    assert rh.exists(cli_run.name, resource_type=rh.Run.RESOURCE_TYPE)


@pytest.mark.rnstest
@pytest.mark.runstest
def test_delete_cli_run_from_rns():
    cli_run = rh.run(CLI_RUN_NAME)
    cli_run.delete_configs()
    assert not rh.exists(name=cli_run.name, resource_type=rh.Run.RESOURCE_TYPE)


# ------------------------- CTX MANAGER RUN ----------------------------------


@pytest.mark.clustertest
@pytest.mark.rnstest
@pytest.mark.runstest
def test_create_local_ctx_manager_run(summer_func, cpu_cluster):
    from runhouse.rh_config import rns_client

    ctx_mgr_func = "my_ctx_mgr_func"

    with rh.Run(path=PATH_TO_CTX_MGR_RUN) as r:
        # Add all Runhouse objects loaded or saved in the context manager to the Run's artifact registry
        # (upstream + downstream artifacts)
        summer_func.save(ctx_mgr_func)

        summer_func(1, 2, run_name="my_new_run")

        current_run = summer_func.system.get_run("my_new_run")
        run_res = current_run.result()
        print(f"Run result: {run_res}")

        cluster = rh.load(name=cpu_cluster.name)
        print(f"Cluster loaded: {cluster.name}")

        summer_func.delete_configs()

    r.save(name=CTX_MGR_RUN)

    print(f"Saved Run with name: {r.name} to path: {r.folder.path}")

    # Artifacts include the rns resolved name (ex: "/jlewitt1/rh-cpu")
    assert r.downstream_artifacts == [
        rns_client.resolve_rns_path(ctx_mgr_func),
        rns_client.resolve_rns_path(cpu_cluster.name),
    ]
    assert r.upstream_artifacts == [
        rns_client.resolve_rns_path(cpu_cluster.name),
    ]


@pytest.mark.localtest
@pytest.mark.runstest
def test_load_named_ctx_manager_run():
    ctx_run = rh.run(path=PATH_TO_CTX_MGR_RUN)
    assert ctx_run.folder.exists_in_system()


@pytest.mark.localtest
@pytest.mark.runstest
def test_read_stdout_from_ctx_manager_run():
    ctx_run = rh.run(path=PATH_TO_CTX_MGR_RUN)
    stdout = ctx_run.stdout()
    pprint(stdout)
    assert stdout


@pytest.mark.rnstest
@pytest.mark.runstest
def test_save_ctx_run_to_rns():
    ctx_run = rh.run(path=PATH_TO_CTX_MGR_RUN)
    ctx_run.save()
    assert rh.exists(name=ctx_run.name, resource_type=rh.Run.RESOURCE_TYPE)


@pytest.mark.clustertest
@pytest.mark.runstest
def test_delete_run_from_system():
    ctx_run = rh.run(path=PATH_TO_CTX_MGR_RUN)
    ctx_run.folder.rm()
    assert not ctx_run.folder.exists_in_system()


@pytest.mark.rnstest
@pytest.mark.runstest
def test_delete_run_from_rns():
    ctx_run = rh.run(CTX_MGR_RUN)
    ctx_run.delete_configs()
    assert not rh.exists(name=ctx_run.name, resource_type=rh.Run.RESOURCE_TYPE)


if __name__ == "__main__":
    unittest.main()
