import runhouse as rh


pip_reqs = ["torch", "numpy"]

# Note: this is currently failing. need to add resource type to rns_server
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
    assert remote_env.req == pip_reqs


def test_create_env():
    test_env = rh.env(name="test_env", reqs=pip_reqs)
    assert len(test_env.reqs) == 2
    assert test_env.reqs == pip_reqs


def test_to_system():
    test_env = rh.env(name="test_env", reqs=["numpy"])
    system = rh.cluster("^rh-cpu").up_if_not()

    test_env.to(system)
    res = system.run_python(["import numpy"])
    assert res[0][0] == 0  # import was successful


def test_function_to_env():
    system = rh.cluster("^rh-cpu").up_if_not()
    test_env = rh.env(name="test_env", reqs=pip_reqs)

    def summer(a, b):
        return a + b

    function = rh.function(summer).to(system, test_env)
    assert set(pip_reqs).issubset(set(function.env.reqs))

    res = system.run_python(["import numpy"])
    assert res[0][0] == 0
