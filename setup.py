from setuptools import setup, find_packages
import os
from os import path
from dotenv import load_dotenv

load_dotenv()

# Model package name
NAME = 'bagged_lightgbm'
# Current Version
VERSION = os.getenv('APP_VERSION', 'latest')

# Dependecies for the package
with open('requirements.txt') as r:
    DEPENDENCIES = [
        dep for dep in map(str.strip, r.readlines())
        if all([not dep.startswith("#"),
                not dep.endswith("#dev"),
                len(dep) > 0])
    ]

# Project descrpition
with open("README.md") as f:
    LONG_DESCRIPTION = f.read()


setup(
    name=NAME,
    version=VERSION,
    description='A simple model to predict apartments rental prices',
    long_description=LONG_DESCRIPTION,
    author='Victor Rabêllo',
    packages=find_packages(exclude=("tests", "docs")),
    entry_points={
        'console_scripts': [
            '{name}={name}.main:cli'.format(name=NAME)
        ],
    },
    # external packages as dependencies
    install_requires=DEPENDENCIES
)