import numpy
from distutils.core import setup
from Cython.Build import cythonize

setup(
    name = "evo-design",
    version = '0.0.0',
    author = 'joelsimon.net',
    author_email='joelsimon6@gmail.com',
    license = 'MIT',
    include_dirs = [numpy.get_include()],
    ext_modules = cythonize("evo_design/*.pyx",
                            include_path = [numpy.get_include()])
)
