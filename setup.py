# -*- coding: utf-8 -*-

from setuptools import setup, find_packages


with open('README.rst') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='fittnuss',
    version='0.1.0',
    description='Framework of Inversion of Tsunami deposits considering Transport of Nonuniform Unsteady Suspension and Sediment entrainment',
    long_description=readme,
    author='Hajime Naruse',
    author_email='naruse@kueps.kyoto-u.ac.jp',
    url='https://github.com/narusehajime/fittnuss',
    license=license,
    install_requires=['numpy','numba', 'scipy', 'configparser', 'matplotlib', 'time', ],
    packages=find_packages(exclude=('tests', 'docs'))
)

