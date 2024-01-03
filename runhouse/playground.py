""" A simple app to generate a list of stargazers for a github repo. """
from pathlib import Path
import requests

import runhouse as rh

# token = Path("~/.github_token").expanduser().read_text().strip() \
#     if Path("~/.github_token").expanduser().exists() else None

token = "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"

def get_stargazers(repo: str, pages=5):
    """ Get the stargazers for a repo. """
    url = f"https://api.github.com/repos/{repo}/stargazers"
    for i in range(int(pages)):
        r = requests.get(url, headers={"Authorization": f"Bearer {token}"} if token else {})
        print(f"status code: {r.status_code}")
        print("result", r.json())
        yield [user["login"] for user in r.json()]
        if "next" in r.links:
            url = r.links["next"]["url"]
        else:
            break


if __name__ == "__main__":
    # cluster = rh.cluster(name="rh-cpu-josh-1",
    #                      instance_type="CPU:2",
    #                      open_ports=[443],
    #                      server_connection_type="tls",
    #                      den_auth=True,
    #                      autostop=-1).up_if_not()

    cluster = rh.cluster(name="rh-cpu-josh-2",
                         instance_type="CPU:2",
                         open_ports=[443],
                         server_connection_type="tls",
                         den_auth=True,
                         autostop=-1).up_if_not()
    
    cluster.run([f"echo {token} > ~/.github_token"])
    stars = rh.fn(get_stargazers, name="get_stargazers").to(cluster)
    for users in stars("run-house/runhouse"):
        print(users)
    stars.share(visibility="unlisted")

    #  curl --cacert '/Users/josh.l/.rh/certs/rh_server.crt' https://44.201.100.159/check