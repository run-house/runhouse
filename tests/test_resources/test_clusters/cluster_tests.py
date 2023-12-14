import copy
import unittest
from pathlib import Path

import pytest

import runhouse as rh
from runhouse.resources.hardware import OnDemandCluster
from runhouse.resources.hardware.utils import ServerConnectionType
from runhouse.rns.utils.api import resolve_absolute_path


def is_on_cluster(cluster):
    return cluster.on_this_cluster()


def np_array(num_list: list):
    import numpy as np

    return np.array(num_list)


def sd_generate_image(prompt):
    from diffusers import StableDiffusionPipeline

    model = StableDiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-2-base"
    ).to("cuda")
    return model(prompt).images[0]


@pytest.mark.clustertest
def test_cluster_config(ondemand_cpu_cluster):
    config = ondemand_cpu_cluster.config_for_rns
    cluster2 = OnDemandCluster.from_config(config)
    assert cluster2.address == ondemand_cpu_cluster.address


@pytest.mark.clustertest
@pytest.mark.rnstest
def test_cluster_sharing(ondemand_cpu_cluster):
    ondemand_cpu_cluster.share(
        users=["donny@run.house", "josh@run.house"],
        access_level="write",
        notify_users=False,
    )
    assert True


@pytest.mark.clustertest
def test_read_shared_cluster(ondemand_cpu_cluster):
    res = ondemand_cpu_cluster.run_python(["import numpy", "print(numpy.__version__)"])
    assert res[0][1]


@pytest.mark.clustertest
def test_install(cluster):
    cluster.install_packages(
        [
            "./",
            "torch==1.12.1",
            # 'conda:jupyterlab',
            # 'gh:pytorch/vision'
        ]
    )


@pytest.mark.clustertest
def test_basic_run(cluster):
    # Create temp file where fn's will be stored
    test_cmd = "echo hi"
    res = cluster.run(commands=[test_cmd])
    assert "hi" in res[0][1]


@pytest.mark.clustertest
def test_restart_server(cluster):
    cluster.up_if_not()
    codes = cluster.restart_server(resync_rh=False)
    assert codes


@pytest.mark.clustertest
def test_on_same_cluster(cluster):
    hw_copy = copy.copy(cluster)

    func_hw = rh.function(is_on_cluster).to(cluster)
    assert func_hw(cluster)
    assert func_hw(hw_copy)


@pytest.mark.clustertest
def test_on_diff_cluster(ondemand_cpu_cluster, byo_cpu):
    func_hw = rh.function(is_on_cluster).to(ondemand_cpu_cluster)
    assert not func_hw(byo_cpu)


@pytest.mark.clustertest
def test_byo_cluster(byo_cpu, local_folder):
    from tests.test_resources.test_modules.test_functions.conftest import summer

    assert byo_cpu.is_up()

    summer_func = rh.function(summer).to(byo_cpu)
    assert summer_func(1, 2) == 3

    byo_cpu.put("test_obj", list(range(10)))
    assert byo_cpu.get("test_obj") == list(range(10))

    local_folder = local_folder.to(byo_cpu)
    assert "sample_file_0.txt" in local_folder.ls(full_paths=False)


@pytest.mark.clustertest
def test_byo_cluster_with_https(byo_cpu):
    tls_connection = ServerConnectionType.TLS.value
    byo_cpu.server_connection_type = tls_connection
    byo_cpu.restart_server()

    assert byo_cpu.server_connection_type == tls_connection

    local_cert_path = byo_cpu.cert_config.cert_path
    assert Path(local_cert_path).exists()

    # Confirm we can send https requests to the cluster
    byo_cpu.install_packages(["numpy"])


@pytest.mark.clustertest
def test_byo_proxy(byo_cpu, local_folder):
    from tests.test_resources.test_modules.test_functions.conftest import summer

    rh.globals.open_cluster_tunnels.pop(byo_cpu.address)
    byo_cpu.client = None
    # byo_cpu._rpc_tunnel.close()
    byo_cpu._rpc_tunnel = None

    byo_cpu._ssh_creds["ssh_host"] = "127.0.0.1"
    byo_cpu._ssh_creds.update(
        {"ssh_proxy_command": "ssh -W %h:%p ubuntu@test-byo-cluster"}
    )
    assert byo_cpu.up_if_not()

    status, stdout, _ = byo_cpu.run(["echo hi"])[0]
    assert status == 0
    assert stdout == "hi\n"

    summer_func = rh.function(summer, env=rh.env(working_dir="local:./")).to(byo_cpu)
    assert summer_func(1, 2) == 3

    byo_cpu.put("test_obj", list(range(10)))
    assert byo_cpu.get("test_obj") == list(range(10))

    # TODO: uncomment out when in-mem lands
    # local_folder = local_folder.to(byo_cpu)
    # assert "sample_file_0.txt" in local_folder.ls(full_paths=False)


@pytest.mark.clustertest
def test_cluster_with_den_auth(ondemand_https_cluster_with_auth, summer_func_with_auth):
    ondemand_https_cluster_with_auth.restart_server()
    from runhouse.globals import configs

    # Create an invalid token, confirm the server does not accept the request
    orig_token = configs.get("token")

    # Request should return 200 using a valid token
    summer_func_with_auth(1, 2)

    configs.set("token", "abcd123")

    try:
        # Request should raise an exception with an invalid token
        summer_func_with_auth(1, 2)
    except ValueError as e:
        assert "Invalid or expired token" in str(e)

    configs.set("token", orig_token)


@pytest.mark.clustertest
def test_start_server_with_custom_certs(
    ondemand_https_cluster_with_auth, summer_func_with_auth
):
    # NOTE: to check certificate matching:
    # openssl x509 -noout -modulus -in rh_server.crt | openssl md5
    # openssl rsa -noout -modulus -in rh_server.key | openssl md5
    from runhouse.servers.http.certs import TLSCertConfig

    ssl_certfile = f"~/ssl/certs/{ondemand_https_cluster_with_auth.name}/rh_server.crt"
    ssl_keyfile = f"~/ssl/private/{ondemand_https_cluster_with_auth.name}/rh_server.key"

    # # NOTE: need to include the IP of the cluster when generating the cert
    TLSCertConfig(
        key_path=ssl_keyfile,
        cert_path=ssl_certfile,
        dir_name=ondemand_https_cluster_with_auth.name,
    ).generate_certs(address=ondemand_https_cluster_with_auth.address)

    # # Restart the server using the custom certs
    ondemand_https_cluster_with_auth.ssl_certfile = ssl_certfile
    ondemand_https_cluster_with_auth.ssl_keyfile = ssl_keyfile
    ondemand_https_cluster_with_auth.restart_server()

    try:
        summer_func_with_auth(1, 2)
    except Exception as e:
        assert False, f"Failed to connect to server with custom certs: {e}"

    Path(resolve_absolute_path(ssl_certfile)).unlink()
    Path(resolve_absolute_path(ssl_keyfile)).unlink()


if __name__ == "__main__":
    unittest.main()
