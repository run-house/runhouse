# script usage: python runhouse/scripts/docs/update_rst.py docs/tutorials/xxx/file.rst

import os
from argparse import ArgumentParser
from pathlib import Path
from typing import List, Tuple


def update_file(
    filename,
    replacements: List[Tuple],
    link_colab=False,
):
    with open(filename, "r") as file:
        # data = file.read()
        data = file.readlines()

    for replacement in replacements:
        data = [line.replace(replacement[0], replacement[1]) for line in data]

    if link_colab:
        ipynb_file = filename.replace("tutorials", "notebooks").replace("rst", "ipynb")
        colab_lines = [
            ".. raw:: html\n",
            "\n",
            f'    <p><a href="https://colab.research.google.com/github/run-house/runhouse/blob/stable/{ipynb_file}">\n'
            '    <img height="20px" width="117px" src="https://colab.research.google.com/assets/colab-badge.svg" \
alt="Open In Colab"/></a></p>\n',
            "\n",
        ]
        data = data[:3] + colab_lines + data[3:]

    with open(filename, "w") as file:
        file.writelines(data)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--files",
        nargs="+",
        default=None,
        help="Paths of rst files to update. If not provided, will apply to all .rst files found in the directory.",
    )
    parser.add_argument(
        "--link-colab",
        action="store_true",
        default=False,
        help="Add colab link subsection under title.",
    )
    args = parser.parse_args()

    files = args.files or Path(os.getcwd()).rglob("*.rst")

    for filename in files:
        replacements = [
            (".. code:: python", ".. code:: ipython3"),
            (
                ".. parsed-literal::\n\n",
                ".. parsed-literal::\n    :class: code-output\n\n",
            ),
        ]

        update_file(filename, replacements, link_colab=args.link_colab)
