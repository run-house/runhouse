import os
import tarfile
import time
from tempfile import TemporaryFile
from kubernetes.client import ApiException
from kubernetes.client.api import core_v1_api

# TODO: this needs work
"""Messing around with k8s stuff - for now starting with EC2 so come back to this"""


def _create_container(api_instance):
    resp = None
    name = os.getenv('K8_NAME')
    namespace = os.getenv('K8_NAMESPACE')
    try:
        resp = api_instance.read_namespaced_pod(name=name,
                                                namespace=namespace)
    except ApiException as e:
        if e.status != 404:
            print("Unknown error: %s" % e)
            exit(1)
    if resp:
        print("Pod already exists")
    else:
        print("Pod %s does not exist. Creating it..." % name)
        # Build a pod that is continuously running
        pod_manifest = {
            'apiVersion': 'v1',
            'kind': 'Pod',
            'metadata': {
                'name': name
            },
            'spec': {
                'containers': [{
                    'image': 'runhouse:1.0.1',
                    'name': 'sleep',
                    "args": [
                        "/bin/sh",
                        "-c",
                        "while true;do date;sleep 5; done"
                    ]
                }]
            }
        }
        resp = api_instance.create_namespaced_pod(body=pod_manifest,
                                                  namespace=namespace)
        while True:
            resp = api_instance.read_namespaced_pod(name=name,
                                                    namespace=namespace)
            if resp.status.phase != 'Pending':
                break
            time.sleep(1)
        print("Done.")


def _configure_instance():
    from kubernetes import config
    config.load_kube_config()
    config.assert_hostname = False  # needed for local dev
    # from kubernetes.client import Configuration
    # c = Configuration()
    # Configuration.set_default(c)
    core_v1 = core_v1_api.CoreV1Api()
    return core_v1


def _copy_to_remote(api_instance, name, namespace):
    from kubernetes.stream import stream

    # exec_command = ['/bin/sh', '-c', cmd]
    exec_command = ['tar', 'xvf', '-', '-C', '/']

    source_file = '/runhouse/cli.py'
    destination_path = '/dev/cli.py'
    print("name", name)
    print("namespace", namespace)
    resp = stream(api_instance.connect_get_namespaced_pod_exec,
                  name,
                  namespace,
                  command=exec_command,
                  stderr=True, stdin=True,
                  stdout=True, tty=False,
                  _preload_content=False)
    print("Resp", resp)
    with TemporaryFile() as tar_buffer:
        with tarfile.open(fileobj=tar_buffer, mode='w') as tar:
            tar.add(name=source_file, arcname=destination_path)
        tar_buffer.seek(0)
        commands = []
        commands.append(tar_buffer.read())
        print("commands", commands)
        while resp.is_open():
            print("resp is open")
            resp.update(timeout=1)
            if resp.peek_stdout():
                print("STDOUT: %s" % resp.read_stdout())
            if resp.peek_stderr():
                print("STDERR: %s" % resp.read_stderr())
            if commands:
                print("commands")
                c = commands.pop(0)
                resp.write_stdin(str(c))
                # works if source and destination files are txt format
                # and the above line is resp.write_stdin(c.decode())
            else:
                break
        resp.close()

    # while resp.is_open():
    #     resp.update(timeout=1)
    #     if resp.peek_stdout():
    #         print("STDOUT: %s" % resp.read_stdout())
    #     if resp.peek_stderr():
    #         print("STDERR: %s" % resp.read_stderr())
    #
    # resp.close()
    #
    # if resp.returncode != 0:
    #     raise Exception("Task scripts execute fail")


api_instance = _configure_instance()
print("api instance:", api_instance)

# Create the pod if it does not exist
_create_container(api_instance=api_instance)
