from setuptools import setup, find_packages
import os

# Read requirements from requirements.txt
with open('requirements.txt') as f:
    requirements = f.read().splitlines()

# Read long description from README.md
with open('README.md', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name="ds_policy",
    version="0.1.0",
    author="Felix",
    description="Dynamical System Policy for robot control",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/felixzheng02/ds_policy",
    packages=['ds_policy'],
    python_requires=">=3.8",
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    include_package_data=True,
) 