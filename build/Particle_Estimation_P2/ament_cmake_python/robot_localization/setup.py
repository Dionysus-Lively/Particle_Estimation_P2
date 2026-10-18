from setuptools import find_packages
from setuptools import setup

setup(
    name='robot_localization',
    version='0.0.0',
    packages=find_packages(
        include=('robot_localization', 'robot_localization.*')),
)
