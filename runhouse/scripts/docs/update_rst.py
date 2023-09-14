# script usage: python runhouse/scripts/docs/update_rst.py docs/tutorials/xxx/file.rst

import os
from argparse import ArgumentParser
from pathlib import Path


def replace_text(filename, original, new):
    with open(filename, "r") as file:
        data = file.read()
        data = data.replace(original, new)

    with open(filename, "w") as file:
        file.write(data)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--files",
        nargs="+",
        default=None,
        help="paths of rst files to update. If not provided, will apply to all .rst files found in the directory.",
    )
    args = parser.parse_args()

    files = args.files or Path(os.getcwd()).rglob("*.rst")

    for filename in files:
        replace_text(filename, ".. code:: python", ".. code:: ipython3")
        replace_text(
            filename,
            ".. parsed-literal::\n\n",
            ".. parsed-literal::\n    :class: code-output\n\n",
        )
