import runhouse as rh


class ModuleContainingMapper:
    def repro(self):

        print("Create remote module...")
        # Broken
        # remote_module = rh.module(ModuleToMap)()

        # Broken
        # remote_module = rh.module(ModuleToMap)().to(self.system, name="module_to_map", env=self.env)

        # Broken
        remote_module = rh.module(ModuleToMap, env=self.env)()

        print("Creating mapper...")
        mapper = rh.mapper(remote_module)
        mapper.add_replicas(3)

        print("Mapping...")
        args = [1, 2, 3]
        mapper.map(args, method="do_something")


class ModuleToMap:
    def do_something(self, a: int):
        return a + 1


if __name__ == "__main__":

    cluster = rh.cluster("rh-cpu", instance_type="CPU:2").save().up_if_not()
    cluster.restart_server()

    env = rh.env(
        name="test_env",
        reqs=["langchain"],
        secrets=["huggingface"],
    )

    remote_module = rh.module(ModuleContainingMapper).to(cluster, env=env)
    remote_module.repro()
