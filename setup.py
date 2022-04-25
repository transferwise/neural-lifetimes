#!/usr/bin/env python

from setuptools import find_packages, setup


def _read_requirements_file(path: str):
    with open(path) as f:
        return list(
            map(
                lambda req: req.strip(),
                f.readlines(),
            )
        )


with open("README.md") as f:
    long_description = f.read()

# TODO Mark: Update before release
setup(
    name="event-model",
    version="0.1.0",
    description="User behavior prediction from event data.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Wise",
    url="https://github.com/transferwise/neural_lifetimes",
    classifiers=[
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    py_modules=["neural_lifetimes"],
    install_requires=_read_requirements_file("requirements.txt"),
    extras_require={
        "test": _read_requirements_file("requirements-dev.txt"),
    },
    packages=find_packages(exclude=["tests*"]),
    package_data={"neural_lifetimes.datasets": ["data/*.pkl"]},
)
