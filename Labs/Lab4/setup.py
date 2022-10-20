from distutils.core import setup
from distutils.extension import Extension

import numpy
from Cython.Build import cythonize

extensions = [
    Extension("src_cy", ["src_cy.pyx"], include_dirs=[numpy.get_include()]),
]

setup(
    name="src_cy",
    ext_modules=cythonize(["src_cy.pyx"], annotate=True, language_level="3"),
    include_dirs=[numpy.get_include()],
)
