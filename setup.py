import pathlib

from setuptools import setup


CWD = pathlib.Path(__file__).absolute().parent


def get_version():
    return "1.0.0"


def get_description():
    return "something"


setup(name="transformer", version=get_version(), long_description=get_description())