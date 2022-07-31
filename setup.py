from setuptools import setup, find_packages
from os import path
import versioneer


here = path.abspath(path.dirname(__file__))

with open(path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()


setup(
    name="torchfitter",
    version=versioneer.get_version(),
    description="Trainer to optimize PyTorch models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    cmdclass=versioneer.get_cmdclass(),
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    author="Alejandro Pérez-Sanjuán",
    python_requires=">=3.7,",
    install_requires=[
        "rich",
        "numpy>=1.20.0",
        "accelerate>=0.11.0",
        "scikit-learn",
        "torchmetrics",
        "torch>=1.1.0",
    ],
)
