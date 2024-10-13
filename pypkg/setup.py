from glob import glob
from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext
from os.path import splitext, basename
from setuptools import setup
from setuptools import find_packages
import glob

def get_pyname(fname):
  return splitext(basename(fname))[0]

print([get_pyname(fname) for fname in sorted(glob.glob('src/*.py')) if not get_pyname(fname) == '__init__'])
print(find_packages(where="src"))

# def setup(**kwargs):
#   print(kwargs)

setup(
    name="imu_relativepose_estimator",
    version="1.1.0",
    author='Hiroya Sato',
    description="The package of the estimator of the IMUs' relative pose.",
    package_dir={'': "src"},
    packages=find_packages(where="src"),
    py_modules=[get_pyname(fname) for fname in sorted(glob.glob('src/*.py')) if not get_pyname(fname) == '__init__'],
    ext_modules=[
        Pybind11Extension(
            'imu_relpose_estim.utils._noisecov_helper',
            ['exts/noisecov_helper.cpp'],
            language='c++',
            extra_compile_args = ['-std=c++11', '-Wall', '-O3'],
        ),
        Pybind11Extension(
            'imu_relpose_estim.utils._hdbingham_helper',
            ['exts/hdbingham_helper.cpp'],
            language='c++',
            extra_compile_args = ['-std=c++11', '-Wall', '-O3'],
        ),
    ],
    install_requires=['numpy', 'pybind11>=2.2'],
    cmdclass={'build_ext': build_ext},
    zip_safe=False,
)
