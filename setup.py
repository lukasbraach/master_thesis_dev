#!/usr/bin/env python

from setuptools import find_packages, setup

setup(
    name="master-thesis",
    version="0.0.1",
    description="Master thesis",
    author="Lukas Braach",
    author_email="lukasbraach@gmail.com",
    url="https://github.com/lukasbraach/master_thesis_dev",
    install_requires=["lightning", "hydra-core"],
    packages=find_packages(),
    # use this to customize global commands available in the terminal after installing the package
    entry_points={
        "console_scripts": [
            "train_command = src.train:main",
            "eval_command = src.eval:main",
        ]
    },
)
