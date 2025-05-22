import kubetorch as kt

# ## Write your code
# This regular Python code is developed locally, and then
# deployed to Kubernetes with Kubetorch. On first execution, it
# may take a little time to allocate compute; subsequently, changes to this function
# will hot sync instantaneously for interactive development. Then, the dispatch
# can be scheduled or put into CI as-is to reach production.
def hello_world(num_prints=1):
    for print_num in range(num_prints):
        print("Hello world ", print_num)


# ## Define compute, deploy, and call
# You define compute with kt.Compute(), and then send the `hello_world`
# function to that compute to run. You can see that you get back a callable
# with the same function signature as the original, and you can call it identically.
if __name__ == "__main__":
    compute = kt.Compute(cpus=1)

    remote_hello = kt.fn(hello_world).to(compute)

    results = remote_hello(5)
