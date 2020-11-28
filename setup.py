NAME="arxiv-generator"
AUTHOR="Paul Baumstark"
VERSION="0.1.0"

import os

from setuptools import setup, find_packages


PATH_ROOT = os.path.dirname(__file__)


def load_requirements(
    path_dir=PATH_ROOT, file_name="requirements.txt", comment_char="#"
):
    with open(os.path.join(path_dir, file_name), "r") as file:
        lines = [ln.strip() for ln in file.readlines()]
    reqs = []
    for ln in lines:
        # filer all comments
        if comment_char in ln:
            ln = ln[: ln.index(comment_char)].strip()
        if ln:  # if requirement is not empty
            reqs.append(ln)
    return reqs

def get_extras_require():

    requirements = {
        "scripts": ["hydra-core"],
        "dashboard": ["streamlit"]
    }

    return requirements


setup(
    name=NAME,
    author=AUTHOR,
    version=VERSION,
    packages=find_packages(),
    install_requires=load_requirements(),
    extras_require=get_extras_require()
)
