#!/usr/bin/env python
# -*- coding: utf-8 -*-
from setuptools import setup, find_packages
import os

CLASSIFIERS = [
    'Intended Audience :: Science/Research',
    'Programming Language :: Python',
    'Programming Language :: JavaScript',
    'Topic :: Internet :: WWW/HTTP :: Dynamic Content',
    'Topic :: Multimedia :: Sound/Audio',
    'Topic :: Multimedia :: Sound/Audio :: Analysis',
    'Topic :: Multimedia :: Sound/Audio :: Players',
    'Topic :: Scientific/Engineering :: Information Analysis', ]

KEYWORDS = 'audio analyze transcode graph player metadata'

setup(
  name = "TimeSide",
  url='http://code.google.com/p/timeside',
  description = "open and fast web audio components",
  long_description = open('README.rst').read(),
  author = ["Guillaume Pellerin", "Olivier Guilyardi", "Riccardo Zaccarelli", "Paul Brossier"],
  author_email = ["yomguy@parisson.com","olivier@samalyse.com", "riccardo.zaccarelli@gmail.com", "piem@piem.org"],
  version = '0.3.2',
  install_requires = [
	    'setuptools',
		'numpy',
		'gst',
  ],
  platforms=['OS Independent'],
  license='Gnu Public License V2',
  classifiers = CLASSIFIERS,
  keywords = KEYWORDS,
  packages = find_packages(),
  include_package_data = True,
  zip_safe = False,
)
