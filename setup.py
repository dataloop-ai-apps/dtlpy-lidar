#! /usr/bin/env python3
# This file is part of Dataloop

"""
    After creating setup.py file run the following commands: (delete the build dir)
    bumpversion patch --allow-dirty
    python setup.py bdist_wheel
"""

from setuptools import setup, find_packages

with open('README.md', encoding="utf8") as f:
    readme = f.read()

with open('requirements.txt') as f:
    requirements = f.read()

packages = [
    package for package in find_packages() if package.startswith('dtlpylidar')
]

setup(name='dtlpylidar',
      classifiers=[
          'Programming Language :: Python :: 3.7',
          'Programming Language :: Python :: 3.8',
          'Programming Language :: Python :: 3.9',
      ],
      version='0.0.3',
      description='Lidar app',
      author='Dataloop Team',
      author_email='shadi.m@dataloop.ai',
      long_description=readme,
      long_description_content_type='text/markdown',
      packages=packages,
      setup_requires=['wheel'],
      install_requires=requirements,
      include_package_data=True
      )
