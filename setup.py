from setuptools import find_packages, setup

setup_requires = []

install_requires = [
    "hifuku",
]

setup(
    name="jsk_table_task_planning",
    version="0.0.0",
    description="thesis",
    author="Hirokazu Ishida",
    author_email="h-ishida@jsk.imi.i.u-tokyo.ac.jp",
    license="MIT",
    install_requires=install_requires,
    packages=find_packages(exclude=("tests", "docs")),
)
