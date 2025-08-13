from pathlib import Path

from setuptools import setup, find_packages

setup(
    name="rkhs",
    version="0.0.1",
    description="A unified framework for marginal and conditional two-sample testing with kernels in JAX.",
    long_description=Path("README.md").read_text(),
    long_description_content_type="text/markdown",
    url="https://github.com/lukashaverbeck/rkhs",
    author="Lukas Haverbeck",
    license="MIT",
    python_requires=">=3.13",
    install_requires=["jax"],
    packages=find_packages(),
    include_package_data=False
)
