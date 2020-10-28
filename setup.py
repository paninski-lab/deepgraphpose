#!/usr/bin/env python
# -*- encoding: utf-8 -*-
from __future__ import absolute_import
from __future__ import print_function

from glob import glob
from os.path import basename
from os.path import splitext

from setuptools import find_packages
from setuptools import setup


setup(
    name="deepgraphpose",
    version="0.1.0",
    description="Deep graph Pose",
    author="Author",
    author_email="email",
    url="url",
    #packages=find_packages('src'),
    packages=['deepgraphpose'],
    package_dir={'':'src'},
    #py_modules=[
    #    splitext(basename(path))[0]
    #    for path in glob("src/deepgraphpose/*.py", recursive=True)
    #],
    include_package_data=True,
    zip_safe=False, #install_requires=['numpy', 'matplotlib']
)
