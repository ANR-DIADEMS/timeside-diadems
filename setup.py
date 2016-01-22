#!/usr/bin/env python
# -*- coding: utf-8 -*-


from setuptools import setup, find_packages
import sys
from setuptools.command.test import test as TestCommand

#try:
#    import multiprocessing  # Workaround for http://bugs.python.org/issue15881
#except ImportError:
#    pass


# Pytest
class PyTest(TestCommand):
    def finalize_options(self):
        TestCommand.finalize_options(self)
        self.test_args = ['tests', '--verbose']
        self.test_suite = True

    def run_tests(self):
        #import here, cause outside the eggs aren't loaded
        import pytest
        errno = pytest.main(self.test_args)
        sys.exit(errno)


CLASSIFIERS = [
    'Intended Audience :: Science/Research',
    'Intended Audience :: Developers',
    'Intended Audience :: Information Technology',
    'Programming Language :: Python',
    'Programming Language :: JavaScript',
    'Topic :: Internet :: WWW/HTTP :: Dynamic Content',
    'Topic :: Multimedia :: Sound/Audio',
    'Topic :: Multimedia :: Sound/Audio :: Analysis',
    'Topic :: Multimedia :: Sound/Audio :: Players',
    'Topic :: Multimedia :: Sound/Audio :: Conversion',
    'Topic :: Scientific/Engineering :: Information Analysis',
    'Topic :: Software Development :: Libraries :: Python Modules',
]

KEYWORDS = 'audio analysis features extraction MIR transcoding graph visualize plot HTML5 interactive metadata player'

setup(
    # Package
    namespace_packages = ['timeside', 'timeside.plugins'],
    packages=find_packages(),
    install_requires=[
        'timeside',
        # Dependencies for limsi_diarization
        'py_sonicvisualiser',
        'pyannote.core',
        'pyannote.features',
        'pyannote.algorithms',
        'pyannote.metrics'
        ],
    # PyPI
    name='TimeSide-Diadems',
    url='https://github.com/ANR-DIADEMS/timeside-diadems',
    description="TimeSide extensions developped during the Diadems project",
    long_description=open('README.rst').read(),
    author="Guillaume Pellerin, Thomas Fillon",
    author_email="yomguy@parisson.com, thomas@parisson.com",
    version='0.1',
    platforms=['OS Independent'],
    license='Gnu Public License V2',
    classifiers=CLASSIFIERS,
    keywords=KEYWORDS,
    include_package_data=True,
    zip_safe=False,
   # Tests
    tests_require=['pytest'],
    cmdclass={'test': PyTest},
    )

