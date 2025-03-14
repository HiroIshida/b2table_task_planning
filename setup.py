from Cython.Build import cythonize
from setuptools import Extension, find_packages, setup

setup_requires = []

install_requires = [
    "hifuku",
]

ext = Extension(
    name="b2table_task_planning.cython._fast_box2d_sd",
    sources=["b2table_task_planning/cython/_fast_box2d_sd.pyx"],
    extra_compile_args=["-O3"],
)

setup(
    name="b2table_task_planning",
    version="0.0.0",
    description="thesis",
    author="Hirokazu Ishida",
    author_email="h-ishida@jsk.imi.i.u-tokyo.ac.jp",
    license="MIT",
    install_requires=install_requires,
    packages=find_packages(exclude=("tests", "docs")),
    ext_modules=cythonize([ext]),
)
