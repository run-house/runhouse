import runhouse as rh


pip_reqs = ["torch", "numpy"]


def test_create_env_from_name_local():
    env_name = "~/local_env"
    local_env = rh.env(name=env_name, reqs=pip_reqs).save()
    del local_env

    remote_env = rh.env(name=env_name)
    assert remote_env.reqs == pip_reqs


def test_create_env_from_name_rns():
    env_name = "rns_env"
    env = rh.env(name=env_name, reqs=pip_reqs).save()
    del env

    remote_env = rh.env(name=env_name)
    assert remote_env.reqs == pip_reqs


def test_create_env():
    test_env = rh.env(name="test_env", reqs=pip_reqs)
    assert len(test_env.reqs) == 2
    assert test_env.reqs == pip_reqs


def test_to_system():
    test_env = rh.env(name="test_env", reqs=["sphinx"])
    system = rh.cluster("^rh-cpu").up_if_not()

    test_env.to(system)
    res = system.run_python(["import sphinx"])
    assert res[0][0] == 0  # import was successful

    system.run(["pip uninstall sphinx -y"])


def test_function_to_env():
    system = rh.cluster("^rh-cpu").up_if_not()
    test_env = rh.env(name="test-env", reqs=["parameterized"]).save()

    def summer(a, b):
        return a + b

    rh.function(summer).to(system, test_env)
    res = system.run_python(["import parameterized"])
    assert res[0][0] == 0

    system.run(["pip uninstall parameterized -y"])
    res = system.run_python(["import parameterized"])
    assert res[0][0] == 1

    rh.function(summer, system=system, env="test-env")
    res = system.run_python(["import parameterized"])
    assert res[0][0] == 0

    system.run(["pip uninstall parameterized -y"])


def test_env_git_reqs():
    system = rh.cluster("^rh-cpu").up_if_not()
    git_package = rh.GitPackage(
        git_url="https://github.com/huggingface/diffusers.git",
        install_method="pip",
        revision="v0.11.1",
    )
    env = rh.env(reqs=[git_package])
    env.to(system)
    res = system.run(["pip freeze | grep diffusers"])
    assert "diffusers" in res[0][1]
    system.run(["pip uninstall diffusers -y"])
