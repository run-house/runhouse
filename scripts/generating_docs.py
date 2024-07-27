import argparse
import os
import shutil
import subprocess
import warnings

import dotenv

import requests

from git import Repo

dotenv.load_dotenv()

GITHUB_TOKEN = os.getenv("GH_TOKEN")

SOURCE_REPO_NAME = "runhouse"
SOURCE_REPO_PATH = f"run-house/{SOURCE_REPO_NAME}"
SOURCE_REPO_URL = f"https://github.com/{SOURCE_REPO_PATH}.git"

TARGET_REPO_NAME = "runhouse-docs"
TARGET_REPO_PATH = f"run-house/{TARGET_REPO_NAME}"

PATH_TO_DOCS = "runhouse/docs/_build"

HEADERS = {
    "Authorization": f"Bearer {GITHUB_TOKEN}",
    "Accept": "application/vnd.github.v3+json",
}


def clone_repo():
    # Check if the repository exists and is a valid git repository
    if os.path.exists(SOURCE_REPO_NAME):
        try:
            repo = Repo(SOURCE_REPO_NAME)
            # Fetch the latest changes
            origin = repo.remotes.origin
            origin.fetch()
        except Exception as e:
            # Handle the case where the directory is not a valid git repo or another issue
            warnings.warn(
                f"Existing directory is not a valid git repository or another error occurred: {e}"
            )
            shutil.rmtree(SOURCE_REPO_NAME)
            Repo.clone_from(SOURCE_REPO_URL, SOURCE_REPO_NAME)
    else:
        # Clone the repository if it does not exist
        Repo.clone_from(SOURCE_REPO_URL, SOURCE_REPO_NAME)


def get_refs_from_repo(url):
    response = requests.get(url, headers=HEADERS)
    refs = response.json()
    return refs


def run_command(command):
    status_codes = subprocess.run(
        command,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return status_codes


def build_and_copy_docs(branch_name, commit_hash=None):
    """Build docs based on the branch's latest commit (or tag revision) in the runhouse repo, and copy the
    resulting JSON files to the runhouse-docs repo."""
    try:
        if commit_hash:
            # Working with a particular release tag
            run_command(f"cd {SOURCE_REPO_NAME} && git reset --hard {commit_hash}")
        else:
            # Make sure the local runhouse folder is up to date with the latest remote branch
            run_command(
                f"cd {SOURCE_REPO_NAME} && git reset --hard origin/{branch_name}"
            )

        # Run the make json command in the "docs" directory of the local Runhouse repo
        res = run_command("cd runhouse && cd docs && make json")
        if res.returncode != 0:
            raise RuntimeError(
                f"Failed to build docs for branch: {branch_name}: {res.stderr}"
            )

        # Get the absolute paths for the source and destination folders
        abs_folder_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), PATH_TO_DOCS)
        )

        # Copy the contents of the generated JSON docs to the runhouse-docs local folder
        shutil.copytree(
            f"{abs_folder_path}/json",
            os.path.abspath(os.path.join(os.path.dirname(__file__), TARGET_REPO_NAME)),
        )

        run_command(
            f"cd {TARGET_REPO_NAME} && git init && git checkout -b {branch_name} "
            f"&& git remote add origin https://github.com/run-house/{TARGET_REPO_NAME}.git "
            f"&& git add . && git commit -m 'Updated docs from script'"
        )

        # Push changes to the remote runhouse-docs repo
        # Overwrite with whatever we have in the remote based on the latest version of the branch in the runhouse repo
        run_command(f"cd {TARGET_REPO_NAME} && git push --force origin {branch_name}")

        # Delete the _build folder in the local file system
        shutil.rmtree(abs_folder_path)

        # Delete the runhouse-docs folder in the local file system
        shutil.rmtree(
            os.path.abspath(os.path.join(os.path.dirname(__file__), TARGET_REPO_NAME))
        )

    except Exception as e:
        raise RuntimeError(f"Failed to build docs for {branch_name}: {str(e)}")


def generate_docs_for_branches():
    branches_url = f"https://api.github.com/repos/{SOURCE_REPO_PATH}/git/refs"
    refs = get_refs_from_repo(branches_url)

    # Build docs and copy JSON files for each branch
    for ref in refs:
        ref_type = ref["ref"].split("/")[-2]
        ref_name = ref["ref"].split("/")[-1]

        if ref_type == "heads":
            print(f"Building docs for branch: {ref_name}")
            # Check out locally to the given branch
            status_codes = run_command(f"cd runhouse && git checkout {ref_name}")
            if status_codes.returncode != 0:
                raise Exception(status_codes.stderr)

            build_and_copy_docs(ref_name)


def generate_docs_for_tag(tag_name):
    """Build docs for a specific release tag."""
    tag_url = (
        f"https://api.github.com/repos/{SOURCE_REPO_PATH}/git/refs/tags/{tag_name}"
    )
    _build_docs_for_tag(tag_url, tag_name)


def generate_docs_for_tags():
    """Build tags for all tagged releases."""
    tags_url = f"https://api.github.com/repos/{SOURCE_REPO_PATH}/releases"
    releases = get_refs_from_repo(tags_url)

    if isinstance(releases, dict) and releases.get("status") != 200:
        raise ValueError(f"Failed to load tags: {releases}")

    for release in releases:
        tag_name = release["tag_name"]
        tag_url = (
            f"https://api.github.com/repos/{SOURCE_REPO_PATH}/git/refs/tags/{tag_name}"
        )
        _build_docs_for_tag(tag_url, tag_name)


def _build_docs_for_tag(tag_url, tag_name):
    tag_response = requests.get(tag_url, headers=HEADERS)
    tag_info = tag_response.json()
    if "status" in tag_info and tag_info.get("status") != 200:
        raise ValueError(f"Failed to get tag info for tag {tag_name}: {tag_info}")

    if "object" in tag_info and "sha" in tag_info["object"]:
        commit_hash = tag_info["object"]["sha"]
        print(f"Building docs for tag: {tag_name} (commit hash: {commit_hash})")
        build_and_copy_docs(tag_name, commit_hash)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate docs and perform related tasks."
    )
    parser.add_argument(
        "--docs-type",
        nargs="?",
        default="tags",
        choices=["branches", "tags", "all"],
        help="Type of docs to build. If 'all' provided will build for both branches and tags. "
        "Default is 'tags'",
    )
    parser.add_argument(
        "--tag-name",
        nargs="?",
        help="Specific release tag name to build docs for. If not specified will build for all --docs-type specified.",
    )

    args = parser.parse_args()
    docs_type = args.docs_type
    tag_name = args.tag_name

    clone_repo()

    if tag_name:
        generate_docs_for_tag(tag_name)
    elif docs_type == "tags":
        generate_docs_for_tags()
    elif docs_type == "branches":
        generate_docs_for_branches()
    elif docs_type == "all":
        # run for all branches & all tags
        generate_docs_for_branches()
        generate_docs_for_tags()
