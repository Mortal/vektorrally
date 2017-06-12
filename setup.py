import os
from setuptools import find_packages, setup

# allow setup.py to be run from any path
os.chdir(os.path.normpath(os.path.dirname(os.path.abspath(__file__))))

with open('README.md') as readme:
    README = readme.read()

setup(
    name='vektorrally',
    version='0.1',
    description='Ipe helpers from vektorrally project',
    long_description=README,
    url='https://github.com/Mortal/vektorrally',
    author='Mathias Rav',
    license='',  # TODO

    # See https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',

        'Intended Audience :: Developers',

        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ],

    packages=find_packages(include=['ipe']),
)
