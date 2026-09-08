"""Packaging setup for mojagg.

Enforces binary platform tags for wheels containing the compiled Mojo shared library.
"""

from setuptools import setup
from setuptools.dist import Distribution


class BinaryDistribution(Distribution):
    def has_ext_modules(self):
        return True


setup(distclass=BinaryDistribution)
