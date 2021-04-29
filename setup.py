from setuptools import setup, find_packages
from os import path
import versioneer


here = path.abspath(path.dirname(__file__))

with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()


setup(
    name='torchfitter',
    version=versioneer.get_version(),
    description='Simple trainer to optimize PyTorch models',
    long_description=long_description,
    long_description_content_type='text/markdown',
    cmdclass=versioneer.get_cmdclass(),
    packages=find_packages(where='src'),
    package_dir={'':'src'},
    author='Alejandro Pérez',
    python_requires='>=3.6,',
    install_requires=[
        "numpy",
        "torch",
        "tqdm",
        "scikit-learn"
    ],
)