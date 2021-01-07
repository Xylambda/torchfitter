from setuptools import setup, find_packages
from os import path
import versioneer


here = path.abspath(path.dirname(__file__))

with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()


setup(
    name='torch fitter',
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    packages=find_packages(where='src'),
    package_dir={'':'src'},
    description='Simple trainer to optimize PyTorch models',
    author='Alejandro Pérez',
    install_requires=[
        "numpy",
        "torch",
        "scikit-learn"
    ],
)