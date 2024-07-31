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

DEFAULT_EMAIL = "josh@run.house"
DEFAULT_USERNAME = "jlewitt1"

SOURCE_REPO_NAME = "runhouse"
SOURCE_REPO_PATH = f"run-house/{SOURCE_REPO_NAME}"
SOURCE_REPO_URL = f"https://github.com/{SOURCE_REPO_PATH}.git"

TARGET_REPO_NAME = "runhouse-docs"
TARGET_REPO_PATH = os.path.abspath(TARGET_REPO_NAME)
PATH_TO_DOCS = "runhouse/docs/_build"

HEADERS = {
    "Authorization": f"Bearer {GITHUB_TOKEN}",
    "Accept": "application/vnd.github.v3+json",
}


def clone_repo():
    if os.path.exists(SOURCE_REPO_NAME):
        try:
            repo = Repo(SOURCE_REPO_NAME)
            origin = repo.remotes.origin
            origin.fetch()
        except Exception as e:
            warnings.warn(
                f"Existing directory is not a valid git repository or another error occurred: {e}"
            )
            shutil.rmtree(SOURCE_REPO_NAME)
            Repo.clone_from(SOURCE_REPO_URL, SOURCE_REPO_NAME)
    else:
        Repo.clone_from(SOURCE_REPO_URL, SOURCE_REPO_NAME)


def get_refs_from_repo(url):
    response = requests.get(url, headers=HEADERS)
    refs = response.json()
    return refs


def run_command(command):
    result = subprocess.run(
        command,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return result


def build_and_copy_docs(branch_name, email, username, commit_hash=None):
    try:
        if not os.path.exists(SOURCE_REPO_NAME):
            raise FileNotFoundError(
                f"The source repository directory '{SOURCE_REPO_NAME}' does not exist."
            )
        if not os.path.exists(os.path.join(SOURCE_REPO_NAME, "docs")):
            raise FileNotFoundError(
                f"The 'docs' directory does not exist in '{SOURCE_REPO_NAME}'."
            )

        if commit_hash:
            run_command(f"cd {SOURCE_REPO_NAME} && git reset --hard {commit_hash}")
        else:
            run_command(
                f"cd {SOURCE_REPO_NAME} && git reset --hard origin/{branch_name}"
            )

        repo_docs_path = os.path.join(SOURCE_REPO_NAME, "docs")
        res = run_command(f"cd {repo_docs_path} && make json")
        if res.returncode != 0:
            raise RuntimeError(
                f"Failed to build docs for branch: {branch_name}: {res.stderr}"
            )

        abs_folder_path = os.path.abspath(os.path.join(repo_docs_path, "_build"))
        json_folder_path = os.path.join(abs_folder_path, "json")

        if not os.path.exists(json_folder_path):
            raise FileNotFoundError(
                f"JSON output directory not found: {json_folder_path}"
            )

        if os.path.exists(TARGET_REPO_PATH):
            shutil.rmtree(TARGET_REPO_PATH)  # Ensure the target path is empty

        shutil.copytree(
            json_folder_path,
            TARGET_REPO_PATH,
            dirs_exist_ok=True,  # Allow copying into an existing directory
        )

        os.chdir(TARGET_REPO_PATH)

        # Configure Git user identity for run-house
        run_command(f'git config --global user.email "{email}"')
        run_command(f'git config --global user.name "{username}"')

        # Set up GitHub token for authentication
        repo_url_with_token = (
            f"https://{GITHUB_TOKEN}@github.com/run-house/{TARGET_REPO_NAME}.git"
        )

        run_command(
            f"git init && git checkout -b {branch_name} "
            f"&& git remote add origin {repo_url_with_token} "
            f"&& git add . && git commit -m 'Updated docs from script'"
        )

        run_command(f"git push --force origin {branch_name}")

        shutil.rmtree(abs_folder_path)
        shutil.rmtree(TARGET_REPO_PATH)

    except Exception as e:
        raise RuntimeError(f"Failed to build docs for {branch_name}: {str(e)}")


def generate_docs_for_branches(email, username):
    branches_url = f"https://api.github.com/repos/{SOURCE_REPO_PATH}/git/refs"
    refs = get_refs_from_repo(branches_url)

    for ref in refs:
        ref_type = ref["ref"].split("/")[-2]
        ref_name = ref["ref"].split("/")[-1]

        if ref_type == "heads":
            print(f"Building docs for branch: {ref_name}")
            run_command(f"cd runhouse && git checkout {ref_name}")
            build_and_copy_docs(ref_name, email=email, username=username)


def generate_docs_for_tag(tag_name, email, username):
    tag_url = (
        f"https://api.github.com/repos/{SOURCE_REPO_PATH}/git/refs/tags/{tag_name}"
    )
    _build_docs_for_tag(tag_url, tag_name, email=email, username=username)


def generate_docs_for_tags(email, username):
    tags_url = f"https://api.github.com/repos/{SOURCE_REPO_PATH}/tags"
    tags: list[dict] = get_refs_from_repo(tags_url)

    if isinstance(tags, dict) and tags.get("status") != 200:
        raise ValueError(f"Failed to load tags: {tags}")

    for tag_metadata in tags:
        tag_name = tag_metadata["name"]
        tag_url = (
            f"https://api.github.com/repos/{SOURCE_REPO_PATH}/git/refs/tags/{tag_name}"
        )
        _build_docs_for_tag(tag_url, tag_name, email=email, username=username)


def _build_docs_for_tag(tag_url, tag_name, email, username):
    tag_response = requests.get(tag_url, headers=HEADERS)
    tag_info = tag_response.json()
    if "status" in tag_info and tag_info.get("status") != 200:
        raise ValueError(f"Failed to get tag info for tag {tag_name}: {tag_info}")

    if "object" in tag_info and "sha" in tag_info["object"]:
        commit_hash = tag_info["object"]["sha"]
        print(f"Building docs for tag: {tag_name} (commit hash: {commit_hash})")
        build_and_copy_docs(
            tag_name, email=email, username=username, commit_hash=commit_hash
        )


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
    parser.add_argument(
        "--username",
        nargs="?",
        default=DEFAULT_USERNAME,
        help="Username to use for authenticating with Github.",
    )
    parser.add_argument(
        "--email",
        default=DEFAULT_EMAIL,
        nargs="?",
        help="Email to use for authenticating with Github.",
    )

    args = parser.parse_args()
    docs_type = args.docs_type
    tag_name = args.tag_name
    email = args.email
    username = args.username

    clone_repo()

    if tag_name:
        generate_docs_for_tag(tag_name, email, username)
    elif docs_type == "tags":
        generate_docs_for_tags(email, username)
    elif docs_type == "branches":
        generate_docs_for_branches(email, username)
    elif docs_type == "all":
        generate_docs_for_branches(email, username)
        generate_docs_for_tags(email, username)
